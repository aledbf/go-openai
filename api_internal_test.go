package openai

import (
	"testing"
)

func TestOpenAIFullURL(t *testing.T) {
	cases := []struct {
		Name   string
		Suffix string
		Expect string
	}{
		{
			"ChatCompletionsURL",
			"/chat/completions",
			"https://api.openai.com/v1/chat/completions",
		},
		{
			"CompletionsURL",
			"/completions",
			"https://api.openai.com/v1/completions",
		},
	}

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			az := DefaultConfig("dummy")
			cli := NewClientWithConfig(az)
			actual := cli.fullURL(c.Suffix)
			if actual != c.Expect {
				t.Errorf("Expected %s, got %s", c.Expect, actual)
			}
			t.Logf("Full URL: %s", actual)
		})
	}
}

func TestRequestAuthHeader(t *testing.T) {
	cases := []struct {
		Name      string
		APIType   APIType
		HeaderKey string
		Token     string
		OrgID     string
		Expect    string
	}{
		{
			"OpenAIDefault",
			"",
			"Authorization",
			"dummy-token-openai",
			"",
			"Bearer dummy-token-openai",
		},
		{
			"OpenAIOrg",
			APITypeOpenAI,
			"Authorization",
			"dummy-token-openai",
			"dummy-org-openai",
			"Bearer dummy-token-openai",
		},
		{
			"OpenAI",
			APITypeOpenAI,
			"Authorization",
			"dummy-token-openai",
			"",
			"Bearer dummy-token-openai",
		},
		{
			"AzureAD",
			APITypeAzureAD,
			"Authorization",
			"dummy-token-azure",
			"",
			"Bearer dummy-token-azure",
		},
		{
			"Azure",
			APITypeAzure,
			AzureAPIKeyHeader,
			"dummy-api-key-here",
			"",
			"dummy-api-key-here",
		},
	}

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			az := DefaultConfig(c.Token)
			az.APIType = c.APIType
			az.OrgID = c.OrgID

			cli := NewClientWithConfig(az)
			req, err := cli.newRequest(t.Context(), "POST", "/chat/completions")
			if err != nil {
				t.Errorf("Failed to create request: %v", err)
			}
			actual := req.Header.Get(c.HeaderKey)
			if actual != c.Expect {
				t.Errorf("Expected %s, got %s", c.Expect, actual)
			}
			t.Logf("%s: %s", c.HeaderKey, actual)
		})
	}
}

func TestAzureFullURL(t *testing.T) {
	cases := []struct {
		Name             string
		BaseURL          string
		AzureModelMapper map[string]string
		Suffix           string
		Model            string
		Expect           string
	}{
		{
			"AzureBaseURLWithSlashAutoStrip",
			"https://httpbin.org/",
			nil,
			"/chat/completions",
			"chatgpt-demo",
			"https://httpbin.org/" +
				"openai/deployments/chatgpt-demo" +
				"/chat/completions?api-version=2023-05-15",
		},
		{
			"AzureBaseURLWithoutSlashOK",
			"https://httpbin.org",
			nil,
			"/chat/completions",
			"chatgpt-demo",
			"https://httpbin.org/" +
				"openai/deployments/chatgpt-demo" +
				"/chat/completions?api-version=2023-05-15",
		},
		{
			"",
			"https://httpbin.org",
			nil,
			"/assistants?limit=10",
			"chatgpt-demo",
			"https://httpbin.org/openai/assistants?api-version=2023-05-15&limit=10",
		},
	}

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			az := DefaultAzureConfig("dummy", c.BaseURL)
			cli := NewClientWithConfig(az)
			// /openai/deployments/{engine}/chat/completions?api-version={api_version}
			actual := cli.fullURL(c.Suffix, withModel(c.Model))
			if actual != c.Expect {
				t.Errorf("Expected %s, got %s", c.Expect, actual)
			}
			t.Logf("Full URL: %s", actual)
		})
	}
}

func TestCloudflareAzureFullURL(t *testing.T) {
	cases := []struct {
		Name    string
		BaseURL string
		Suffix  string
		Expect  string
	}{
		{
			"CloudflareAzureBaseURLWithSlashAutoStrip",
			"https://gateway.ai.cloudflare.com/v1/dnekeim2i39dmm4mldemakiem3i4mkw3/demo/azure-openai/resource/chatgpt-demo/",
			"/chat/completions",
			"https://gateway.ai.cloudflare.com/v1/dnekeim2i39dmm4mldemakiem3i4mkw3/demo/azure-openai/resource/chatgpt-demo/" +
				"chat/completions?api-version=2023-05-15",
		},
		{
			"",
			"https://gateway.ai.cloudflare.com/v1/dnekeim2i39dmm4mldemakiem3i4mkw3/demo/azure-openai/resource/chatgpt-demo",
			"/assistants?limit=10",
			"https://gateway.ai.cloudflare.com/v1/dnekeim2i39dmm4mldemakiem3i4mkw3/demo/azure-openai/resource/chatgpt-demo" +
				"/assistants?api-version=2023-05-15&limit=10",
		},
	}

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			az := DefaultAzureConfig("dummy", c.BaseURL)
			az.APIType = APITypeCloudflareAzure

			cli := NewClientWithConfig(az)

			actual := cli.fullURL(c.Suffix)
			if actual != c.Expect {
				t.Errorf("Expected %s, got %s", c.Expect, actual)
			}
			t.Logf("Full URL: %s", actual)
		})
	}
}
