from experiments.eswa_20261019.monitor import attention_required
from experiments.eswa_20261019.monitor import queue_attention


def test_queue_completion_and_pending_gate_need_followthrough():
    assert queue_attention('formal_input_preparation_running',False)
    assert not queue_attention('formal_input_preparation_running',True)
    assert queue_attention('engineering_complete_requires_freeze',False)


def test_retained_terminal_rows_ignore_reused_pid():
    assert not attention_required('FAILED_RETAINED',True,5000)
    assert not attention_required('COMPLETED',True,5000)
    assert attention_required('FAILED',False,0)
    assert attention_required('PROCESS_EXITED_CHECK_RESULT',False,0)
    assert attention_required('RUNNING',True,1201)
    assert not attention_required('RUNNING',True,100)
