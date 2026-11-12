//! Streaming stage: an `http_body::Body` wrapper over a translated Anthropic
//! Messages SSE stream that collects the `tool_use` blocks and holds the final
//! `message_delta` (the one carrying `stop_reason`) until the classifier has
//! answered, then stamps `safeguard_results` into its `delta`.
//!
//! Everything else is forwarded immediately and in order (events are decoded
//! and re-encoded canonically: `event: <name>\ndata: <json>\n\n`). Tool-use
//! ids are never rewritten. If the upstream already answered (native leg, or
//! a provider that implements the protocol), the stream is passed through and
//! the local classifier does not run. In shadow mode nothing is held: the
//! evaluation runs detached for metrics only.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use agent_llm::types::safeguards::{
	MessageDeltaCheck, SafeguardResults, SafeguardsReviewPlan, StreamedToolUses,
	inspect_message_delta, stamp_message_delta,
};
use bytes::{Bytes, BytesMut};
use http_body::Frame;
use pin_project_lite::pin_project;
use serde_json::Value;
use tokio_sse_codec::{Event, Frame as SseFrame, SseDecoder};
use tokio_util::codec::Decoder;
use tracing::warn;

use super::{EvalContext, Mode, evaluate};

type EvalFuture = Pin<Box<dyn Future<Output = SafeguardResults> + Send + 'static>>;

enum State {
	/// Decode upstream frames and forward them.
	Streaming,
	/// The final `message_delta` is held while the classifier answers.
	Evaluating {
		fut: EvalFuture,
		held: Value,
		held_name: String,
	},
	/// Upstream exhausted; flush queued output then finish.
	Draining,
	Done,
}

pin_project! {
	pub struct SafeguardsSseBody {
		#[pin]
		inner: agent_http::RawBody,
		plan: Arc<SafeguardsReviewPlan>,
		ctx: EvalContext,
		decoder: SseDecoder<Bytes>,
		decode_buffer: BytesMut,
		buffer_limit: usize,
		streamed: StreamedToolUses,
		out: VecDeque<Bytes>,
		state: State,
		// Set once results exist for this message (ours or the upstream's).
		answered: bool,
	}
}

impl SafeguardsSseBody {
	#[allow(clippy::new_ret_no_self)]
	pub fn new(
		inner: agent_http::RawBody,
		plan: Arc<SafeguardsReviewPlan>,
		ctx: EvalContext,
		buffer_limit: usize,
	) -> agent_http::RawBody {
		agent_http::RawBody::new(Self {
			inner,
			plan,
			ctx,
			decoder: SseDecoder::with_max_size(buffer_limit),
			decode_buffer: BytesMut::new(),
			buffer_limit,
			streamed: StreamedToolUses::default(),
			out: VecDeque::new(),
			state: State::Streaming,
			answered: false,
		})
	}
}

fn encode_frame(frame: SseFrame<Bytes>) -> Bytes {
	match frame {
		SseFrame::Event(Event { name, data, id }) => {
			let mut out = BytesMut::new();
			if let Some(id) = id
				&& !id.is_empty()
			{
				out.extend_from_slice(b"id: ");
				out.extend_from_slice(id.as_bytes());
				out.extend_from_slice(b"\n");
			}
			// The decoder reports the SSE default type ("message") for frames
			// without an `event:` line; Anthropic never uses that name.
			let name_bytes = if name.is_empty() || name == "message" {
				""
			} else {
				name.as_ref()
			};
			out.extend_from_slice(&agent_llm::parse::encode_sse_event(name_bytes, data));
			out.freeze()
		},
		SseFrame::Comment(comment) => {
			let mut out = BytesMut::from(&b":"[..]);
			out.extend_from_slice(&comment);
			out.extend_from_slice(b"\n\n");
			out.freeze()
		},
		SseFrame::Retry(duration) => Bytes::from(format!("retry: {}\n\n", duration.as_millis())),
	}
}

fn encode_json_event(name: &str, event: &Value) -> Bytes {
	let data = serde_json::to_vec(event).unwrap_or_default();
	agent_llm::parse::encode_sse_event(name, Bytes::from(data))
}

impl SafeguardsSseBody {
	/// Decode everything currently buffered. Returns `true` when an evaluation
	/// was started (state switched to `Evaluating`); decoding stops there so the
	/// held event stays the last decoded one.
	#[allow(clippy::too_many_arguments)]
	fn drain_decoder(
		decoder: &mut SseDecoder<Bytes>,
		buffer: &mut BytesMut,
		buffer_limit: usize,
		streamed: &mut StreamedToolUses,
		out: &mut VecDeque<Bytes>,
		answered: &mut bool,
		plan: &Arc<SafeguardsReviewPlan>,
		ctx: &EvalContext,
	) -> Option<State> {
		loop {
			match decoder.decode(buffer) {
				Ok(Some(SseFrame::Event(event))) => {
					let name = event.name.to_string();
					let data = event.data.clone();
					if data.as_ref() == b"[DONE]" {
						out.push_back(encode_frame(SseFrame::Event(event)));
						continue;
					}
					let parsed: Option<Value> = serde_json::from_slice(&data).ok();
					let Some(json) = parsed else {
						out.push_back(encode_frame(SseFrame::Event(event)));
						continue;
					};
					streamed.observe(&json);
					match inspect_message_delta(&json) {
						MessageDeltaCheck::PassThrough => {
							out.push_back(encode_frame(SseFrame::Event(event)));
						},
						MessageDeltaCheck::AlreadyAnswered => {
							*answered = true;
							super::record_decision_label(ctx.metrics.as_deref(), "upstream_answered");
							out.push_back(encode_frame(SseFrame::Event(event)));
						},
						MessageDeltaCheck::NeedsResults { hit_max_tokens } if !*answered => {
							*answered = true;
							let pending = streamed.finish(hit_max_tokens);
							let plan = plan.clone();
							let ctx_owned = ctx.clone();
							if ctx.mode == Mode::Shadow {
								tokio::spawn(async move {
									let _ = evaluate(&plan, &pending, &ctx_owned).await;
								});
								out.push_back(encode_frame(SseFrame::Event(event)));
								continue;
							}
							let fut: EvalFuture =
								Box::pin(async move { evaluate(&plan, &pending, &ctx_owned).await });
							return Some(State::Evaluating {
								fut,
								held: json,
								held_name: name,
							});
						},
						MessageDeltaCheck::NeedsResults { .. } => {
							// A second final delta in one message: already answered.
							out.push_back(encode_frame(SseFrame::Event(event)));
						},
					}
				},
				Ok(Some(other)) => out.push_back(encode_frame(other)),
				Ok(None) => return None,
				Err(error) => {
					// Forward the undecodable bytes as-is and resynchronize.
					warn!("safeguards: SSE decode error, forwarding raw bytes: {error}");
					if !buffer.is_empty() {
						out.push_back(buffer.split().freeze());
					}
					*decoder = SseDecoder::with_max_size(buffer_limit);
					return None;
				},
			}
		}
	}
}

impl http_body::Body for SafeguardsSseBody {
	type Data = Bytes;
	type Error = axum_core::Error;

	fn poll_frame(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
		let mut this = self.project();
		loop {
			// Always flush queued output first so ordering is preserved.
			if let Some(bytes) = this.out.pop_front() {
				return Poll::Ready(Some(Ok(Frame::data(bytes))));
			}
			match this.state {
				State::Done => return Poll::Ready(None),
				State::Draining => {
					*this.state = State::Done;
					return Poll::Ready(None);
				},
				State::Evaluating {
					fut,
					held,
					held_name,
				} => match fut.as_mut().poll(cx) {
					Poll::Pending => return Poll::Pending,
					Poll::Ready(results) => {
						let mut event = std::mem::take(held);
						let name = std::mem::take(held_name);
						if !stamp_message_delta(&mut event, &results) {
							warn!("safeguards: could not stamp message_delta; forwarding unstamped");
						}
						this.out.push_back(encode_json_event(&name, &event));
						*this.state = State::Streaming;
						// Anything that arrived after the held event (e.g. message_stop)
						// may already sit in the decode buffer; drain it now.
						if let Some(next) = Self::drain_decoder(
							this.decoder,
							this.decode_buffer,
							*this.buffer_limit,
							this.streamed,
							this.out,
							this.answered,
							this.plan,
							this.ctx,
						) {
							*this.state = next;
						}
					},
				},
				State::Streaming => match this.inner.as_mut().poll_frame(cx) {
					Poll::Pending => return Poll::Pending,
					Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
					Poll::Ready(None) => {
						// Final partial frame without a trailing blank line, if any.
						if !this.decode_buffer.is_empty() {
							if let Ok(Some(frame)) = this.decoder.decode_eof(this.decode_buffer) {
								this.out.push_back(encode_frame(frame));
							}
							if !this.decode_buffer.is_empty() {
								this.out.push_back(this.decode_buffer.split().freeze());
							}
						}
						*this.state = State::Draining;
					},
					Poll::Ready(Some(Ok(frame))) => {
						let Some(data) = frame.data_ref() else {
							// Trailers pass through untouched.
							return Poll::Ready(Some(Ok(frame)));
						};
						this.decode_buffer.extend_from_slice(data);
						if let Some(next) = Self::drain_decoder(
							this.decoder,
							this.decode_buffer,
							*this.buffer_limit,
							this.streamed,
							this.out,
							this.answered,
							this.plan,
							this.ctx,
						) {
							*this.state = next;
						}
					},
				},
			}
		}
	}
}

impl SafeguardsSseBody {
	/// Whether the stream has produced results (for tests).
	pub fn answered(&self) -> bool {
		self.answered
	}
}
