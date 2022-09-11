Traces/Launch scripts can be found here (branch awsvm): https://github.com/jovonho/mlperf-traces/tree/awsvm

I had to limit most of the traces to 6 fields (max number of arguments bpftrace takes for printf) becasue I was getting too many interleaving lines when using more.

TRACES:

================================================
trace_bio.out
Uses the two kernel probes blk_account_io_start() and blk_account_io_done() to trace block IO requests.
-------------------------------------------------------------------
TIMESTAMP: 		nsecs since boot
TIME:			local time
PID: 			process pid
COMM: 			process name
DONEPID:		PID of process in whose context the blk_account_io_done probe executes. (See note 1)
DONECOMM: 		name of process with above PID
DISK: 			always sda since I filtered for it.
T: 			    request type. W = Write, R = Read.
SECTOR: 		disk sector requested (__sector field of struct request in blk_account_io_start call)
BYTES: 			size of requested data (__data_len field of struct request in blk_account_io_start call)
LAT(ns):		difference btw timestamp of blk_account_io_start and blk_account_io_done calls

Note 1: In Brendan Gregg's book BPF Performance Tools we read the following at the end of 9.3.2:
"The blk_account_io_start() function often fires in process context and occurs when the I/O is queued. Later events,
such as issuing the I/O to the device and I/O completion, may or may not happen in process context, so you cannot rely
on the value of the pid and comm builtins at those later times. The solution is to store them in BPF maps during
blk_account_io_start(), keyed by the request ID, so that they can be retrieved later."
This hints that even the start probe does not always start in process context, which was observed during other experiments.
Ref: https://books.google.ca/books?id=ihTADwAAQBAJ&pg=PT517&lpg=PT517&dq=kprobe+blk+account+io+start&source=bl&ots=bzNO7FBELg&sig=ACfU3U01-3q4b5y3gileFgY7YvY8mJcCdQ&hl=en&sa=X&ved=2ahUKEwiEvu2m9Oj1AhU2jYkEHeVlBR4Q6AF6BAgJEAM#v=onepage&q&f=false


================================================
trace_openat.out

traces calls to the openat() syscall using the sys_enter_openat() and sys_exit_openat() tracepoints.
We also intercept the associated vfs_open() calls here to get the filename in case the truncated path retrieved from openat() probe hides it.
-------------------------------------------------------------------
TIMESTAMP:		nsecs since boot
TIME:			local time
PID:			process pid
FD:             The returned file descriptor (negative if error)
LAT (ns)		call latency in nsecs (difference btw the timestamps of enter and exit tracepoints or kprobe and kretprobe)
FULLPATH        full path of filename, truncated to 63 chars
FILENAME		opened file (note 4)

Note 2: We only need trace openat() and not the open() syscall because "In modern Linux systems (glibc >= 2.26)
the open wrapper always calls the openat syscall."
(src: Lesson 3 - https://github.com/iovisor/bpftrace/blob/master/docs/tutorial_one_liners.md)


================================================
trace_read.out
Traces calls to read() syscalls using tracepoints.
-------------------------------------------------------------------
TIMESTAMP: 		nsecs since boot
PID:			process pid
FD:			    file descriptor
RET (B):		number of bytes read (return value of syscall)
LAT (ns):		timsetamp difference between enter and exit probes
FILE:			filename

================================================
trace_write.out
Traces calls to write() syscalls using tracepoints. Also intercepts vfs_write() calls to get the filename.
Due to less write activity, I was able to included more fields here in multiple printf calls without getting interleaving lines.
-------------------------------------------------------------------
TIMESTAMP: 		nsecs since boot
PID:			process pid
FD:			    file descriptor
BUFFER ADDR:    address at which the write was issued
OFFSET:         offset from the buffer address at which write was issued
REQ (B):		given size of the write (count argument of the syscall)
RET (B):		number of bytes written (return value of syscall)
LAT (ns):		timsetamp difference between enter and exit tracepoints
FILENAME:       filename

================================================
dlrm.log
Application log of the workload (uses mlperf_logging module). Has UNIX timestamps.

================================================
gpu.out
Output of `nvidia-smi pmon` showing GPU usage metrics every second.
Column definition is not 100% certain as the output of pmon is not precisely detailed in the ref.
Ref: https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
-------------------------------------------------------------------
Date:		YYYYMMDD
Time:		HH:MM:SS
gpu:		index of GPU (0 to 7)
pid:		process using the GPU
type: 		Type of the process. C=Compute, G=Graphics. Always C in this case.
sm:			% of time over the past sample period during which one or more kernels was executing on the GPU. (SM = Streaming Multiprocessor = GPU processor) 
mem:		% of time over the past sample period during which global (device) memory was being read or written. 
enc:  		% of time over the past sample period during which the GPU's video encoder was being used.
dec:  		% of time over the past sample period during which the GPU's video decoder was being used.
fb:			Frame Buffer memory usage in MB.
command:    name of program


================================================
cpu.out
Output of mpstat showing CPU usage metrics every second.
Ref: https://linux.die.net/man/1/mpstat
-------------------------------------------------------------------
Time:		HH:MM:SS
CPU:		CPU index or 'all' for average values
%usr:		% of CPU use that occurred while executing at the user level (application)
%nice: 		% of CPU use that occurred while executing at the user level with nice priority.
%sys:		% of CPU use that occurred while executing at the system level (kernel). This does not include time spent servicing hardware and software interrupts.
%iowait:	% of time that the CPU or CPUs were idle during which the system had an outstanding disk I/O request.
%irq:		% of time spent by the CPU or CPUs to service hardware interrupts.
%soft:		% of time spent by the CPU or CPUs to service software interrupts.
%steal:		% of time spent in involuntary wait by the virtual CPU or CPUs while the hypervisor was servicing another virtual processor.
%guest:		% of time spent by the CPU or CPUs to run a virtual processor.
%gnice:		% of time spent by the CPU or CPUs to run a niced guest.
%idle:		% of time that the CPU or CPUs were idle and the system did not have an outstanding disk I/O request.


================================================
trace_time_align.out

Trace used to estimate the nsecs since boot timestamp of a given local time and align the bpf traces to the app log and cpu/gpu traces.



Additional traces, not used for the graphs

================================================
strace.out

Output of strace with the following exclude list: 
'trace=!ioctl,clock_gettime,sched_yield,nanosleep,sched_getaffinity,sched_setaffinity,futex,set_robust_list'

Contains TIDs instead of PIDs

================================================
pids_tids.out
Output of `ps aux | grep python` 2min into training, so misses short lived threads.
Maps TIDs in strace to PIDs used in the other traces


================================================
trace_close.out
Traces calls to close() syscalls using tracepoints. 
-------------------------------------------------------------------
TIMESTAMP: 		nsecs since boot
PID:			process id
FD:             file descriptor associated if file-backed mmap
LAT (ns):		timsetamp difference between enter and exit tracepoints
FILENAME:       filename if a file-backed mmap

================================================
trace_create_del.out
Traces calls to mkdir(), rmdir() and unlink() syscalls using tracepoints. 
-------------------------------------------------------------------
mkdir():
TIMESTAMP         PID   FUNCTION   MODE    LAT(ns)  FILENAME

rmdir():
TIMESTAMP         PID    FUNCTION  LAT(ns)  FILENAME

unlink():
TIMESTAMP         PID    FUNCTION   LAT(ns)  FILENAME FULLPATH

================================================
trace_mmap.out
Traces calls to mmap() and munmap() syscalls using tracepoints. 
-------------------------------------------------------------------
For mmap():
TIMESTAMP: 		nsecs since boot
TID:			actually the pid
FD:             file descriptor associated if file-backed mmap
ADDRESS:		hint address given to mmap()
LAT (ns):		timsetamp difference between enter and exit tracepoints
FILENAME:       filename if a file-backed mmap

For munmap():
TIMESTAMP: 		nsecs since boot
TID:			actually the pid
ADDRESS:		hint address given to mmap()
LEN:       length of regiong to unmap
LAT (ns):		timsetamp difference between enter and exit tracepoints


