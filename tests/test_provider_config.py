import os
from unittest.mock import patch
import pytest
from brain.config import load_config
from brain.llm import LLM

@pytest.mark.parametrize('env,expected', [
    ({'OPENAI_API_KEY':'test-only'},'openai'),
    ({'DATABRICKS_TOKEN':'test-only'},'openai'),
    ({'ANTHROPIC_API_KEY':'test-only'},'anthropic'),
    ({'BRAIN_PROVIDER':'anthropic','OPENAI_API_KEY':'test-only'},'anthropic'),
    ({'BRAIN_PROVIDER':' OpenAI ','OPENAI_API_KEY':'test-only'},'openai'),
    ({'BRAIN_PROVIDER':'mock'},'mock'),
])
def test_provider_uses_configured_credentials(tmp_path,env,expected):
    with patch.dict(os.environ,dict(env,BRAIN_HOME=str(tmp_path)),clear=True), patch('brain.config._load_dotenv'):
        cfg=load_config()
    assert cfg.provider==expected
    assert cfg.model.startswith({'openai':'gpt-','anthropic':'claude-','mock':'mock'}[expected])

def test_missing_key_reported_before_sdk_import(tmp_path):
    with patch.dict(os.environ,{'BRAIN_HOME':str(tmp_path),'BRAIN_PROVIDER':'anthropic'},clear=True), patch('brain.config._load_dotenv'):
        cfg=load_config()
    with pytest.raises(RuntimeError,match='ANTHROPIC_API_KEY'):
        LLM(cfg)
