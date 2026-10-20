from shortcutminer.protocol.model_runtime import ModelUsage
from experiments.eswa_20261019.usage_budget import ReservedUsageBudget


def budget(cap=10000):
    return ReservedUsageBudget(max_text_calls=5,max_vlm_calls=5,max_vlm_images=5,
        max_text_per_run=5,max_vlm_per_run=5,max_vlm_images_per_run=5,max_tokens=cap)


def test_timeout_usage_is_reserved_not_zero_or_known_billing():
    b=budget();assert b.reserve('r','text_agent',0)
    b.begin_call();b.account_request({'messages':[{'role':'user','content':'hello'}],'max_tokens':128})
    b.observe_usage(ModelUsage(available=False))
    assert b.usage_incomplete and b.known_tokens==0 and b.reserved_unknown_tokens>128
    assert b.reserve('r','text_agent',0)
    b.max_tokens=b.reserved_unknown_tokens
    assert not b.reserve('r','text_agent',0)


def test_unknown_multimodal_usage_is_still_fail_closed():
    b=budget();b.begin_call()
    b.account_request({'messages':[{'role':'user','content':[{'type':'image_url'}]}],'max_tokens':128})
    b.observe_usage(ModelUsage(available=False))
    assert b.unknown_unbounded and not b.reserve('r','text_agent',0)


def test_reported_usage_is_counted_once_and_call_limits_remain():
    b=budget();b.begin_call()
    b.observe_usage(ModelUsage(available=True,prompt_tokens=10,completion_tokens=5,total_tokens=15))
    assert b.known_tokens==15 and b.reserved_unknown_tokens==0 and not b.usage_incomplete
    for _ in range(5):assert b.reserve('r','text_agent',0)
    assert not b.reserve('r','text_agent',0)


def test_streaming_transport_selection_is_preserved():
    from experiments.eswa_20261019.usage_budget import observed_transport
    from shortcutminer.runtime.openai_compatible import UrllibStreamingJsonTransport
    assert isinstance(observed_transport(UrllibStreamingJsonTransport(),budget()),UrllibStreamingJsonTransport)
