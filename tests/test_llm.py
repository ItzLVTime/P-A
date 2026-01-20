import pytest
import os
from unittest.mock import patch, MagicMock
from backend.llm import ask_llm, get_client, LLMError


class TestLLM:
    def test_missing_api_base_raises_error(self):
        """Test that missing API base raises an error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError, match="LITELLM_API_BASE not set"):
                get_client()

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises an error."""
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://localhost:4000"}, clear=True):
            with pytest.raises(LLMError, match="LITELLM_API_KEY not set"):
                get_client()

    def test_ask_llm_with_mocked_client(self):
        """Test ask_llm with a mocked OpenAI client."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

        with patch.dict(os.environ, {
            "LITELLM_API_BASE": "http://localhost:4000",
            "LITELLM_API_KEY": "test-key"
        }):
            with patch("backend.llm.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = ask_llm("Test prompt")

                assert result == "Hello!"
                mock_client.chat.completions.create.assert_called_once()

    def test_ask_llm_uses_env_model(self):
        """Test that ask_llm uses model from environment."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]

        with patch.dict(os.environ, {
            "LITELLM_API_BASE": "http://localhost:4000",
            "LITELLM_API_KEY": "test-key",
            "LITELLM_MODEL": "custom-model"
        }):
            with patch("backend.llm.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                ask_llm("Test")

                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs["model"] == "custom-model"
