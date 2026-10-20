"""Opt-in bounded handling of missing usage on failed plain-text requests.

Unavailable usage is never recorded as zero or as actual billed usage. Image
requests with unavailable usage remain fail-closed. Existing clients are unchanged.
"""
import hashlib
import json
from threading import local
from shortcutminer.runtime.openai_compatible import PilotInvocationBudgetState,UrllibStreamingJsonTransport


class ReservedUsageBudget(PilotInvocationBudgetState):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._request=local();self.unknown_token_reservations=[];self.unknown_unbounded=False

    def begin_call(self):self._request.estimate=None

    def account_request(self,body):
        messages=body.get('messages',[])
        if not all(isinstance(m.get('content'),str) for m in messages):
            self._request.estimate=None;return
        encoded=json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode('utf-8')
        self._request.estimate={'request_messages_sha256':hashlib.sha256(encoded).hexdigest(),
            'reserved_tokens':len(encoded)+int(body.get('max_tokens',0))+2048,
            'method':'UTF8 message byte count + maximum output tokens + 2048 overhead; conservative estimate, not provider billing'}

    @property
    def reserved_unknown_tokens(self):return sum(r['reserved_tokens'] for r in self.unknown_token_reservations)

    def observe_usage(self,usage):
        with self.lock:
            if usage.available:self.known_tokens+=usage.total_tokens or 0
            else:
                self.usage_incomplete=True
                estimate=getattr(self._request,'estimate',None)
                if estimate is None:self.unknown_unbounded=True
                else:self.unknown_token_reservations.append(dict(estimate))

    def reserve(self,run_id,runtime_kind,images,*,restore=False):
        with self.lock:
            counts=self.by_run.setdefault(run_id,{'text':0,'vlm':0,'images':0})
            if not restore and self.max_tokens is not None and (
                self.unknown_unbounded or self.known_tokens+self.reserved_unknown_tokens>=self.max_tokens):return False
            if runtime_kind=='semantic_vlm':
                if (self.vlm_calls>=self.max_vlm_calls or counts['vlm']>=self.max_vlm_per_run
                    or self.vlm_images+images>self.max_vlm_images
                    or counts['images']+images>self.max_vlm_images_per_run):return False
                self.vlm_calls+=1;self.vlm_images+=images;counts['vlm']+=1;counts['images']+=images
            else:
                if self.text_calls>=self.max_text_calls or counts['text']>=self.max_text_per_run:return False
                self.text_calls+=1;counts['text']+=1
            return True


class ObservedTransport:
    def __init__(self,delegate,budget):self.delegate=delegate;self.budget=budget
    def post_json(self,url,*,headers,body,timeout):
        self.budget.account_request(body)
        return self.delegate.post_json(url,headers=headers,body=body,timeout=timeout)


class ObservedStreamingTransport(ObservedTransport,UrllibStreamingJsonTransport):
    """Keep the existing client's isinstance-based streaming envelope selection."""


def observed_transport(delegate,budget):
    cls=ObservedStreamingTransport if isinstance(delegate,UrllibStreamingJsonTransport) else ObservedTransport
    return cls(delegate,budget)
