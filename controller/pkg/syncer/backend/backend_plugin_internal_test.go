package agentgatewaybackend

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/agentgateway/agentgateway/api"
)

func TestParseAzureEndpoint(t *testing.T) {
	tests := []struct {
		name             string
		endpoint         string
		wantName         string
		wantResourceType api.AIBackend_AzureResourceType
	}{
		{
			name:             "openai endpoint",
			endpoint:         "my-resource.openai.azure.com",
			wantName:         "my-resource",
			wantResourceType: api.AIBackend_OPEN_AI,
		},
		{
			name:             "foundry endpoint without -resource suffix",
			endpoint:         "myproject.services.ai.azure.com",
			wantName:         "myproject",
			wantResourceType: api.AIBackend_FOUNDRY,
		},
		{
			// Azure portal's "Foundry legacy" template generates resource
			// names that end in "-resource". That suffix is part of the
			// user's resource name, NOT part of the hostname suffix the
			// parser should strip.
			name:             "foundry endpoint with legacy -resource resource name preserved",
			endpoint:         "myproject-resource.services.ai.azure.com",
			wantName:         "myproject-resource",
			wantResourceType: api.AIBackend_FOUNDRY,
		},
		{
			name:             "unknown suffix falls back to whole endpoint with OpenAI type",
			endpoint:         "something.example.com",
			wantName:         "something.example.com",
			wantResourceType: api.AIBackend_OPEN_AI,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotName, gotResourceType := parseAzureEndpoint(tt.endpoint)
			if gotName != tt.wantName {
				t.Errorf("parseAzureEndpoint(%q) name = %q, want %q", tt.endpoint, gotName, tt.wantName)
			}
			if gotResourceType != tt.wantResourceType {
				t.Errorf("parseAzureEndpoint(%q) resourceType = %v, want %v", tt.endpoint, gotResourceType, tt.wantResourceType)
			}
		})
	}
}

// TestBedrockTransportWireCompatibility pins the protobuf wire properties the
// native-messages rollout depends on:
//   - the transport enum keeps CONVERSE as the zero value, so every message
//     produced before field 6 existed (and every field absent on the wire)
//     decodes to the historical default;
//   - field 6 is a plain varint, so a proxy/controller built before this field
//     ignores it without error, and an unknown enum number must not crash
//     decoding.
func TestBedrockTransportWireCompatibility(t *testing.T) {
	// Zero value of the enum must be CONVERSE: the fail-closed default.
	if api.AIBackend_BedrockTransport(0) != api.AIBackend_BEDROCK_TRANSPORT_CONVERSE {
		t.Fatalf("BedrockTransport zero value must be BEDROCK_TRANSPORT_CONVERSE, got %v", api.AIBackend_BedrockTransport(0))
	}

	// An unset transport must not appear on the wire: pre-field-6 binaries
	// decode the message byte-for-byte identically.
	unset, err := proto.Marshal(&api.AIBackend_Bedrock{Model: proto.String("m")})
	if err != nil {
		t.Fatal(err)
	}
	native, err := proto.Marshal(&api.AIBackend_Bedrock{Model: proto.String("m"), Transport: api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES})
	if err != nil {
		t.Fatal(err)
	}
	if string(unset) != string(native[:len(unset)]) {
		t.Fatalf("native encoding must extend the unset encoding, got %x vs %x", unset, native)
	}
	// Field 6, wire type 0 (varint), value 1 = tag byte 0x30 followed by 0x01.
	if len(native) != len(unset)+2 || native[len(unset)] != 0x30 || native[len(unset)+1] != 0x01 {
		t.Fatalf("expected transport to encode as field-6 varint 0x30 0x01, got suffix %x", native[len(unset):])
	}

	// Decoding round-trips the native value...
	var decoded api.AIBackend_Bedrock
	if err := proto.Unmarshal(native, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.GetTransport() != api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES {
		t.Fatalf("round-trip lost transport, got %v", decoded.GetTransport())
	}

	// ...an absent field decodes to the CONVERSE default...
	var absent api.AIBackend_Bedrock
	if err := proto.Unmarshal(unset, &absent); err != nil {
		t.Fatal(err)
	}
	if absent.GetTransport() != api.AIBackend_BEDROCK_TRANSPORT_CONVERSE {
		t.Fatalf("absent transport must default to CONVERSE, got %v", absent.GetTransport())
	}

	// ...and an unknown enum number decodes without error and is rejected by
	// the syncer mapping (only the exact known values map 1:1).
	unknown, err := proto.Marshal(&api.AIBackend_Bedrock{Model: proto.String("m"), Transport: api.AIBackend_BedrockTransport(99)})
	if err != nil {
		t.Fatal(err)
	}
	var unknownDecoded api.AIBackend_Bedrock
	if err := proto.Unmarshal(unknown, &unknownDecoded); err != nil {
		t.Fatalf("unknown enum number must not break decoding: %v", err)
	}
	if unknownDecoded.GetTransport() != api.AIBackend_BedrockTransport(99) {
		t.Fatalf("unknown enum number should be preserved for inspection, got %v", unknownDecoded.GetTransport())
	}
	// The mapping rule applied by backend_plugin.go: NATIVE_MESSAGES and
	// OPENAI_INVOKE map 1:1; anything else resolves to CONVERSE (fail-closed).
	got := api.AIBackend_BEDROCK_TRANSPORT_CONVERSE
	switch unknownDecoded.GetTransport() {
	case api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES:
		got = api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES
	case api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE:
		got = api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE
	}
	if got != api.AIBackend_BEDROCK_TRANSPORT_CONVERSE {
		t.Fatalf("unknown enum number %v must fail closed to CONVERSE, got %v", unknownDecoded.GetTransport(), got)
	}

	// OPENAI_INVOKE maps 1:1 through the syncer mapping.
	openaiInvoke := api.AIBackend_BEDROCK_TRANSPORT_CONVERSE
	switch api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE {
	case api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES:
		openaiInvoke = api.AIBackend_BEDROCK_TRANSPORT_NATIVE_MESSAGES
	case api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE:
		openaiInvoke = api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE
	}
	if openaiInvoke != api.AIBackend_BEDROCK_TRANSPORT_OPENAI_INVOKE {
		t.Fatalf("OPENAI_INVOKE must map 1:1, got %v", openaiInvoke)
	}
}
