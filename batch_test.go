package openai_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test/checks"
)

func TestUploadBatchFile(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/files", handleCreateFile)
	req := openai.UploadBatchFileRequest{}
	req.AddChatCompletion("req-1", openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	_, err := client.UploadBatchFile(t.Context(), req)
	checks.NoError(t, err, "UploadBatchFile error")
}

func TestCreateBatch(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/batches", handleBatchEndpoint)
	_, err := client.CreateBatch(t.Context(), openai.CreateBatchRequest{
		InputFileID:      "file-abc",
		Endpoint:         openai.BatchEndpointChatCompletions,
		CompletionWindow: "24h",
	})
	checks.NoError(t, err, "CreateBatch error")
}

func TestCreateBatchWithUploadFile(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/files", handleCreateFile)
	server.RegisterHandler("/v1/batches", handleBatchEndpoint)
	req := openai.CreateBatchWithUploadFileRequest{
		Endpoint: openai.BatchEndpointChatCompletions,
	}
	req.AddChatCompletion("req-1", openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	_, err := client.CreateBatchWithUploadFile(t.Context(), req)
	checks.NoError(t, err, "CreateBatchWithUploadFile error")
}

func TestRetrieveBatch(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/batches/file-id-1", handleRetrieveBatchEndpoint)
	_, err := client.RetrieveBatch(t.Context(), "file-id-1")
	checks.NoError(t, err, "RetrieveBatch error")
}

func TestCancelBatch(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/batches/file-id-1/cancel", handleCancelBatchEndpoint)
	_, err := client.CancelBatch(t.Context(), "file-id-1")
	checks.NoError(t, err, "RetrieveBatch error")
}

func TestListBatch(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/batches", handleBatchEndpoint)
	after := "batch_abc123"
	limit := 10
	_, err := client.ListBatch(t.Context(), &after, &limit)
	checks.NoError(t, err, "RetrieveBatch error")
}

func TestUploadBatchFileRequest_AddChatCompletion(t *testing.T) {
	type args struct {
		customerID string
		body       openai.ChatCompletionRequest
	}
	tests := []struct {
		name string
		args []args
	}{
		{"", []args{
			{
				customerID: "req-1",
				body: openai.ChatCompletionRequest{
					MaxTokens: 5,
					Model:     openai.GPT3Dot5Turbo,
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    openai.ChatMessageRoleUser,
							Content: "Hello!",
						},
					},
				},
			},
			{
				customerID: "req-2",
				body: openai.ChatCompletionRequest{
					MaxTokens: 5,
					Model:     openai.GPT3Dot5Turbo,
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    openai.ChatMessageRoleUser,
							Content: "Hello!",
						},
					},
				},
			},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &openai.UploadBatchFileRequest{}
			for _, arg := range tt.args {
				r.AddChatCompletion(arg.customerID, arg.body)
			}

			// Compare objects after unmarshalling to avoid field order issues
			got := r.MarshalJSONL()
			gotLines := bytes.Split(got, []byte("\n"))

			// Verify each line can be marshaled back to the expected structure
			if len(gotLines) != len(tt.args) {
				t.Errorf("Expected %d lines, got %d", len(tt.args), len(gotLines))
				return
			}

			for i, line := range gotLines {
				var gotReq openai.BatchChatCompletionRequest
				if err := json.Unmarshal(line, &gotReq); err != nil {
					t.Errorf("Failed to unmarshal line %d: %v", i, err)
					continue
				}

				expectedArg := tt.args[i]
				if gotReq.CustomID != expectedArg.customerID {
					t.Errorf("Line %d: custom_id mismatch: got %q, want %q", i, gotReq.CustomID, expectedArg.customerID)
				}

				if !reflect.DeepEqual(gotReq.Body, expectedArg.body) {
					t.Errorf("Line %d: body mismatch: got %v, want %v", i, gotReq.Body, expectedArg.body)
				}

				if gotReq.Method != "POST" {
					t.Errorf("Line %d: method mismatch: got %q, want %q", i, gotReq.Method, "POST")
				}

				if gotReq.URL != openai.BatchEndpointChatCompletions {
					t.Errorf("Line %d: URL mismatch: got %q, want %q", i, gotReq.URL, openai.BatchEndpointChatCompletions)
				}
			}
		})
	}
}

func TestUploadBatchFileRequest_AddCompletion(t *testing.T) {
	type args struct {
		customerID string
		body       openai.CompletionRequest
	}
	tests := []struct {
		name string
		args []args
	}{
		{"", []args{
			{
				customerID: "req-1",
				body: openai.CompletionRequest{
					Model: openai.GPT3Dot5Turbo,
					User:  "Hello",
				},
			},
			{
				customerID: "req-2",
				body: openai.CompletionRequest{
					Model: openai.GPT3Dot5Turbo,
					User:  "Hello",
				},
			},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &openai.UploadBatchFileRequest{}
			for _, arg := range tt.args {
				r.AddCompletion(arg.customerID, arg.body)
			}

			// Compare objects after unmarshalling to avoid field order issues
			got := r.MarshalJSONL()
			gotLines := bytes.Split(got, []byte("\n"))

			// Verify each line can be marshaled back to the expected structure
			if len(gotLines) != len(tt.args) {
				t.Errorf("Expected %d lines, got %d", len(tt.args), len(gotLines))
				return
			}

			for i, line := range gotLines {
				var gotReq openai.BatchCompletionRequest
				if err := json.Unmarshal(line, &gotReq); err != nil {
					t.Errorf("Failed to unmarshal line %d: %v", i, err)
					continue
				}

				expectedArg := tt.args[i]
				if gotReq.CustomID != expectedArg.customerID {
					t.Errorf("Line %d: custom_id mismatch: got %q, want %q", i, gotReq.CustomID, expectedArg.customerID)
				}

				if !reflect.DeepEqual(gotReq.Body, expectedArg.body) {
					t.Errorf("Line %d: body mismatch: got %v, want %v", i, gotReq.Body, expectedArg.body)
				}

				if gotReq.Method != "POST" {
					t.Errorf("Line %d: method mismatch: got %q, want %q", i, gotReq.Method, "POST")
				}

				if gotReq.URL != openai.BatchEndpointCompletions {
					t.Errorf("Line %d: URL mismatch: got %q, want %q", i, gotReq.URL, openai.BatchEndpointCompletions)
				}
			}
		})
	}
}

func TestUploadBatchFileRequest_AddEmbedding(t *testing.T) {
	type args struct {
		customerID string
		body       openai.EmbeddingRequest
	}
	tests := []struct {
		name string
		args []args
	}{
		{"", []args{
			{
				customerID: "req-1",
				body: openai.EmbeddingRequest{
					Model: openai.GPT3Dot5Turbo,
					Input: []string{"Hello", "World"},
				},
			},
			{
				customerID: "req-2",
				body: openai.EmbeddingRequest{
					Model: openai.AdaEmbeddingV2,
					Input: []string{"Hello", "World"},
				},
			},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &openai.UploadBatchFileRequest{}
			for _, arg := range tt.args {
				r.AddEmbedding(arg.customerID, arg.body)
			}

			// Compare objects after unmarshalling to avoid field order issues
			got := r.MarshalJSONL()
			gotLines := bytes.Split(got, []byte("\n"))

			// Verify each line can be marshaled back to the expected structure
			if len(gotLines) != len(tt.args) {
				t.Errorf("Expected %d lines, got %d", len(tt.args), len(gotLines))
				return
			}

			for i, line := range gotLines {
				var gotReq openai.BatchEmbeddingRequest
				if err := json.Unmarshal(line, &gotReq); err != nil {
					t.Errorf("Failed to unmarshal line %d: %v", i, err)
					continue
				}

				expectedArg := tt.args[i]
				if gotReq.CustomID != expectedArg.customerID {
					t.Errorf("Line %d: custom_id mismatch: got %q, want %q", i, gotReq.CustomID, expectedArg.customerID)
				}

				// Check the model
				if gotReq.Body.Model != expectedArg.body.Model {
					t.Errorf("Line %d: model mismatch: got %q, want %q", i, gotReq.Body.Model, expectedArg.body.Model)
				}

				// Marshal and unmarshal the input field to compare
				expectedInputJSON, err := json.Marshal(expectedArg.body.Input)
				if err != nil {
					t.Errorf("Failed to marshal expected input: %v", err)
					continue
				}

				gotInputJSON, err := json.Marshal(gotReq.Body.Input)
				if err != nil {
					t.Errorf("Failed to marshal got input: %v", err)
					continue
				}

				if string(gotInputJSON) != string(expectedInputJSON) {
					t.Errorf("Line %d: input mismatch: got %s, want %s", i, gotInputJSON, expectedInputJSON)
				}

				if gotReq.Method != "POST" {
					t.Errorf("Line %d: method mismatch: got %q, want %q", i, gotReq.Method, "POST")
				}

				if gotReq.URL != openai.BatchEndpointEmbeddings {
					t.Errorf("Line %d: URL mismatch: got %q, want %q", i, gotReq.URL, openai.BatchEndpointEmbeddings)
				}
			}
		})
	}
}

func handleBatchEndpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		_, _ = fmt.Fprintln(w, `{
			  "id": "batch_abc123",
			  "object": "batch",
			  "endpoint": "/v1/completions",
			  "errors": null,
			  "input_file_id": "file-abc123",
			  "completion_window": "24h",
			  "status": "completed",
			  "output_file_id": "file-cvaTdG",
			  "error_file_id": "file-HOWS94",
			  "created_at": 1711471533,
			  "in_progress_at": 1711471538,
			  "expires_at": 1711557933,
			  "finalizing_at": 1711493133,
			  "completed_at": 1711493163,
			  "failed_at": null,
			  "expired_at": null,
			  "cancelling_at": null,
			  "cancelled_at": null,
			  "request_counts": {
				"total": 100,
				"completed": 95,
				"failed": 5
			  },
			  "metadata": {
				"customer_id": "user_123456789",
				"batch_description": "Nightly eval job"
			  }
			}`)
	} else if r.Method == http.MethodGet {
		_, _ = fmt.Fprintln(w, `{
			  "object": "list",
			  "data": [
				{
				  "id": "batch_abc123",
				  "object": "batch",
				  "endpoint": "/v1/chat/completions",
				  "errors": null,
				  "input_file_id": "file-abc123",
				  "completion_window": "24h",
				  "status": "completed",
				  "output_file_id": "file-cvaTdG",
				  "error_file_id": "file-HOWS94",
				  "created_at": 1711471533,
				  "in_progress_at": 1711471538,
				  "expires_at": 1711557933,
				  "finalizing_at": 1711493133,
				  "completed_at": 1711493163,
				  "failed_at": null,
				  "expired_at": null,
				  "cancelling_at": null,
				  "cancelled_at": null,
				  "request_counts": {
					"total": 100,
					"completed": 95,
					"failed": 5
				  },
				  "metadata": {
					"customer_id": "user_123456789",
					"batch_description": "Nightly job"
				  }
				}
			  ],
			  "first_id": "batch_abc123",
			  "last_id": "batch_abc456",
			  "has_more": true
			}`)
	}
}

func handleRetrieveBatchEndpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		_, _ = fmt.Fprintln(w, `{
		  "id": "batch_abc123",
		  "object": "batch",
		  "endpoint": "/v1/completions",
		  "errors": null,
		  "input_file_id": "file-abc123",
		  "completion_window": "24h",
		  "status": "completed",
		  "output_file_id": "file-cvaTdG",
		  "error_file_id": "file-HOWS94",
		  "created_at": 1711471533,
		  "in_progress_at": 1711471538,
		  "expires_at": 1711557933,
		  "finalizing_at": 1711493133,
		  "completed_at": 1711493163,
		  "failed_at": null,
		  "expired_at": null,
		  "cancelling_at": null,
		  "cancelled_at": null,
		  "request_counts": {
			"total": 100,
			"completed": 95,
			"failed": 5
		  },
		  "metadata": {
			"customer_id": "user_123456789",
			"batch_description": "Nightly eval job"
		  }
		}`)
	}
}

func handleCancelBatchEndpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		_, _ = fmt.Fprintln(w, `{
		  "id": "batch_abc123",
		  "object": "batch",
		  "endpoint": "/v1/chat/completions",
		  "errors": null,
		  "input_file_id": "file-abc123",
		  "completion_window": "24h",
		  "status": "cancelling",
		  "output_file_id": null,
		  "error_file_id": null,
		  "created_at": 1711471533,
		  "in_progress_at": 1711471538,
		  "expires_at": 1711557933,
		  "finalizing_at": null,
		  "completed_at": null,
		  "failed_at": null,
		  "expired_at": null,
		  "cancelling_at": 1711475133,
		  "cancelled_at": null,
		  "request_counts": {
			"total": 100,
			"completed": 23,
			"failed": 1
		  },
		  "metadata": {
			"customer_id": "user_123456789",
			"batch_description": "Nightly eval job"
		  }
		}`)
	}
}
