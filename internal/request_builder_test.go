package openai //nolint:testpackage // testing private field

import (
	"bytes"
	"errors"
	"net/http"
	"reflect"
	"testing"
)

var errTestMarshallerFailed = errors.New("test marshaller failed")

type failingMarshaller struct{}

func (*failingMarshaller) Marshal(_ any) ([]byte, error) {
	return []byte{}, errTestMarshallerFailed
}

func (*failingMarshaller) Unmarshal(_ []byte, _ any) error {
	return errTestMarshallerFailed
}

func TestRequestBuilderReturnsMarshallerErrors(t *testing.T) {
	builder := HTTPRequestBuilder{
		marshaller: &failingMarshaller{},
	}

	_, err := builder.Build(t.Context(), &Request{
		Method:       http.MethodGet,
		URL:          "/foo",
		Body:         struct{}{},
		Header:       nil,
		ExtraHeaders: nil,
		ExtraQuery:   nil,
	})
	if !errors.Is(err, errTestMarshallerFailed) {
		t.Fatalf("Did not return error when marshaller failed: %v", err)
	}
}

func TestRequestBuilderReturnsRequest(t *testing.T) {
	b := NewRequestBuilder()
	var (
		ctx         = t.Context()
		method      = http.MethodPost
		url         = "/foo"
		request     = map[string]string{"foo": "bar"}
		reqBytes, _ = b.marshaller.Marshal(request)
		want, _     = http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(reqBytes))
	)
	got, _ := b.Build(ctx, &Request{
		Method:       method,
		URL:          url,
		Body:         request,
		Header:       nil,
		ExtraHeaders: nil,
		ExtraQuery:   nil,
	})
	if !reflect.DeepEqual(got.Body, want.Body) ||
		!reflect.DeepEqual(got.URL, want.URL) ||
		!reflect.DeepEqual(got.Method, want.Method) {
		t.Errorf("Build() got = %v, want %v", got, want)
	}
}

func TestRequestBuilderReturnsRequestWhenRequestOfArgsIsNil(t *testing.T) {
	var (
		ctx     = t.Context()
		method  = http.MethodGet
		url     = "/foo"
		want, _ = http.NewRequestWithContext(ctx, method, url, nil)
	)
	b := NewRequestBuilder()
	got, _ := b.Build(ctx, &Request{
		Method:       method,
		URL:          url,
		Body:         nil,
		Header:       nil,
		ExtraHeaders: nil,
		ExtraQuery:   nil,
	})
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Build() got = %v, want %v", got, want)
	}
}
