package openai_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test/checks"
)

// TestGetEngine Tests the retrieve engine endpoint of the API using the mocked server.
func TestGetEngine(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/engines/text-davinci-003", func(w http.ResponseWriter, _ *http.Request) {
		resBytes, _ := json.Marshal(openai.Engine{})
		fmt.Fprintln(w, string(resBytes))
	})
	_, err := client.GetEngine(t.Context(), "text-davinci-003")
	checks.NoError(t, err, "GetEngine error")
}

// TestListEngines Tests the list engines endpoint of the API using the mocked server.
func TestListEngines(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/engines", func(w http.ResponseWriter, _ *http.Request) {
		resBytes, _ := json.Marshal(openai.EnginesList{})
		fmt.Fprintln(w, string(resBytes))
	})
	_, err := client.ListEngines(t.Context())
	checks.NoError(t, err, "ListEngines error")
}

func TestListEnginesReturnError(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/engines", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	})

	_, err := client.ListEngines(t.Context())
	checks.HasError(t, err, "ListEngines did not fail")
}
