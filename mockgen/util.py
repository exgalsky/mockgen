import sys

MockgenDefaults = {
    "ID"   : "Mockgen Default ID",
    "N"    : 512,
    "seed" : 13579,
    "ityp" : "delta"
}

def parprint(*args,**kwargs):
    print("".join(map(str,args)),**kwargs);  sys.stdout.flush()

def profiletime(task_tag, step, times, comm=None, mpiproc=0):
    if comm is not None:
        comm.Barrier()

    dt = time() - times['t0']
    if step in times.keys():
        times[step] += dt
    else:
        times[step] = dt
    times['t0'] = time()

    if mpiproc!=0:
        return times

    if task_tag is not None:
        parprint(f'{task_tag}: {dt:.6f} sec for {step}')
    else:
        parprint(f'{dt:.6f} sec for {step}')
    parprint("")

    return times