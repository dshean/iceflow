#! /usr/bin/env python

#TODO: better error handling, throw/catch exceptions at each stage

#Run ASP's dem_mosaic tool to create a mosaic
#TODO: add threads, all stat types
#TODO: check to make sure dem_mosaic executable is path
def run_dem_mosaic(fn_list, mos_fn=None, stat=None):
    if mos_fn is None:
        mos_fn = 'mos.tif'
    if os.path.exists(mos_fn):
        #Should ask to overwrite
        n = 1
        ok = False
        while not ok:
            if '_%i.tif' % n in mos_fn:
                n += 1
            else:
                ok = True
        mos_fn = os.path.splitext(mos_fn)[0]+'_%i.tif' % n
    outprefix = os.path.splitext(mos_fn)[0]
    cmd = ['dem_mosaic', '-o', outprefix]
    #first, last, min, max, mean, stddev, median, count
    if stat is 'count':
        statflag = '--count'
    elif stat is 'stddev':
        statflag = '--stddev'
    else:
        #Use default weigthed mean
        statflag = ''
    cmd.append(statflag)
    cmd.extend(fn_list)
    print(cmd)
    subprocess.call(cmd)
    out_fn = outprefix+'-tile-0.tif'
    os.rename(out_fn, mos_fn)
    return mos_fn

#TODO: cleanup intermediate files
def run_pc_align(ref_fn, src_fn, cleanup=False, **kwargs):
    #cmd = ['pc_align', ref_dem]
    #Default output name
    out_fn = os.path.splitext(src_fn)[0]+'_trans.tif'
    log_fn = os.path.split(src_fn)[0]+'.log'
    cmd = ['glacierhack_pcalign.sh', ref_fn, src_fn]
    cmd.extend(kwargs)
    cmd.extend(['|', 'tee', log_fn]
    print(cmd)
    subprocess.call(cmd)
    #Note: the glacierhack_pcalign.sh script writes log file from pc_align using tee

    #If we call pc_align directly, need to output log file
    #http://stackoverflow.com/questions/13332268/python-subprocess-command-with-pipe 
    #ps = subprocess.Popen(('ps', '-A'), stdout=subprocess.PIPE)
    #output = subprocess.check_output(('grep', 'process_name'), stdin=ps.stdout)
    #ps.wait()

    #http://stackoverflow.com/questions/4856583/how-do-i-pipe-a-subprocess-call-in-python-to-a-text-file
    #log = open(log_fn, 'w')
    #subprocess.call(cmd, stdout=log)
    #log = None

    if not os.path.exists(out_fn):
        print("Output file not found. Something went wrong")
        print("See pc_align log file: %s" % log_fn)
        out_fn = None
    return out_fn

def pc_align_all(fn_list, ref_fn=None):
    #If we don't have a trusted reference data source
    if ref_fn is None:
        ref_fn = run_dem_mosaic(fn_list) 
    if not os.path.exists(ref_fn):
        #Should be better about throwing exceptions here
        sys.exit("Unable to find specified reference file: %s" % ref_fn)
    for fn in fn_list:
        if fn == ref_fn:
            out_fn_list.append(fn)
        else:
            run_pc_align(ref_fn, fn)
    return out_fn_list
