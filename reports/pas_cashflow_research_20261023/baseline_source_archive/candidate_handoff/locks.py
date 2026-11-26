import contextlib,fcntl,os
from pathlib import Path


class WriterBusy(RuntimeError):pass


class WriterLease:
    """Same three inode locks as the old scored/practice runners."""
    def __init__(self,root):
        root=Path(root)
        self.paths=[root/'practice_live/live_runtime/process.lock',
                    root/'.runtime/pas_competition_live/process.lock',root/'.runtime/wallet_writer.lock']
        self.stack=None;self.pid=None

    def __enter__(self):
        if self.stack is not None:raise RuntimeError('Lease already active')
        stack=contextlib.ExitStack()
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True,exist_ok=True)
                f=stack.enter_context(path.open('a+'))
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            stack.close();raise WriterBusy('Existing wallet writer owns the shared lock') from None
        except BaseException:
            stack.close();raise
        self.stack=stack;self.pid=os.getpid();return self

    def require(self):
        if self.stack is None or self.pid!=os.getpid():raise WriterBusy('Current process has no exclusive writer lease')

    def __exit__(self,*args):
        self.stack.close();self.stack=None;self.pid=None
