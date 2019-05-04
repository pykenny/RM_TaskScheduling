"""
    Structure:
        T: task period
        C: Computation time

"""
import math
from itertools import islice
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def util_bound(n):
    """ Function used to compute utilization bound.

        Args:
            n (int) - Number of tasks to be scheduled.
        Returns:
            (float) Utilization oundary
    """
    return n * (pow(2.0, 1/n) - 1.0)

""" Note for possible settings:
    "CtSw": Context Switch
"""

def util(series, i, settings, beta=0.0):
    """ Function to compute utilization rate
    """
    task_results = []
    dl_delta = series[i]["T"] - series[i]["D"] if "D" in settings and "D" in series[i] else 0.0
    cost_ctsw = settings.get("CtSw", 0.)

    for task in islice(series, 0, i):
        task_util = (task["C"] + 2 * cost_ctsw) / task["T"]
        task_results.append(task_util)
    # Handle current task's case (inflation with deadline restriction)
    task_util = (series[i]["C"] + 2 * cost_ctsw + dl_delta + beta) / series[i]["T"]
    task_results.append(task_util)

    return sum(task_results)


_TASK_TABLE_HEAD = "\n  Task |    T    |    C    |    D    |    B    |   beta   |    C'    |"
_TASK_TABLE_SEPR = "-------+---------+---------+---------+---------+----------+----------+"
_TASK_CONTENT_FT = "   {:02d}  |  {:5.2f}  |  {:5.2f}  |  {:5.2f}  |  {:5.2f}  |  {:5.2f}   |  {:5.2f}   |"
_TASK_TABLE_ENDL = "----------------------------------------------------------------------\n"
def print_task_table(series, settings, betas):
    """ Simple task information table.
    """
    cost_ctsw = settings.get("CtSw", 0.)
    print(_TASK_TABLE_HEAD)
    for idx, task in enumerate(series, 1):
        print(_TASK_TABLE_SEPR)
        deadline = task["D" if "D" in settings and "D" in task else "T"]
        blocking = task.get("B", 0.)
        print(_TASK_CONTENT_FT.format(
            idx, task["T"], task["C"], deadline, blocking, betas[idx-1], task["C"] + 2 * cost_ctsw))
    print(_TASK_TABLE_ENDL)

def print_resource_table(table, col_names):
    """ (TODO) Simple task resource request table.
    """
    return


_TASK_HEAD_TITLE_FS = "*** Schedulability of T_{:d} ***"
_TASK_ANALYSIS_TITLE1_FS = "== Part 1: Utility =="
_TASK_ANALYSIS_TITLE2_FS = "\n== Part 2: Real Analysis =="
_PART1_UTILITY_CPR_FS = "  Utility = {0:.4f} {1:s} {2:.4f}."
_PART1_UTILITY_PASS = "  Pass utility test."
_PART1_UTILITY_FAIL = "  Failed on utility test."
_PART2_R0_FS = "                        --> R_00 = {:.4f}"
_PART2_IT_FS = "  # Iteration {:d} #"
_PART2_BS_FS = "    Base: C_{:d} = {:.4f}"
_PART2_DL_FS = "    Delay from task {:d} = {:.4f}"
_PART2_RN_FS = "                        --> R_{:02d} = {:.4f}"
_PART2_PASS = "  Converge. Schedulable.\n"
_PART2_FAIL = "  Over deadline. Unschedulable.\n"

_POLICIES = ["PIP", "SRP"]

""" Note for resource obtaining definition
    {[Resource_name]: [Obtain_time], ...}
"""

def util_analysis(series, settings):
    """ Whole analysis process
    """
    n_tasks = len(series)
    tasks_sorted = sorted(series, key=lambda x: x["T"])
    cost_ctsw = settings.get("CtSw", 0.)

    # Decide beta
    if "Plcy" in settings and settings["Plcy"] in _POLICIES:
        # Build up policy table
        # Step 1: Collect possible resources
        vtr = DictVectorizer()
        resource_dicts = [s.get("Res", {}) for s in tasks_sorted]
        resource_table = vtr.fit_transform(resource_dicts).toarray()
        resource_mapping = vtr.vocabulary_
        print_resource_table(resource_table, vtr.feature_names_)
        ceil = {}
        for res_name, res_idx in resource_mapping.items():
            ceil[res_name] = np.flatnonzero(resource_table[::, res_idx])[0]

        if settings["Plcy"] == _POLICIES[0]: # PIP
            beta_task = []
            beta_sect = []
            for t_idx, task in enumerate(tasks_sorted):
                # Obtain resources that can block off this task
                over_res_name = [res_name for res_name, res_ceil in ceil.items() if res_ceil <= t_idx]
                over_res_id = [resource_mapping[res_name] for res_name in over_res_name]
                # Select resources (col) that can block this process,
                # then take row max for lower-prior tasks and sum up
                # Note: It's safe to sum up empty nparray
                beta_task.append(resource_table[t_idx+1::, over_res_id].max(axis=1, initial=0.).sum())
                beta_sect.append(resource_table[t_idx+1::, over_res_id].max(axis=0, initial=0.).sum())
            betas = np.array([beta_task.append, beta_sect.append]).min(axis=0)
        elif settings["Plcy"] == _POLICIES[1]: # SRP
            betas = []
            for t_idx, task in enumerate(tasks_sorted):
                over_res_name = [res_name for res_name, res_ceil in ceil.items() if res_ceil <= t_idx]
                over_res_id = [resource_mapping[res_name] for res_name in over_res_name]
                betas.append(resource_table[t_idx+1::, over_res_id].max(initial=0.))
    else:
        betas = [
            max((s.get("B", 0.0) for s in tasks_sorted[i+1:]), default=0.0) if "B" in settings else 0.0
            for i in range(n_tasks)
        ]

    print_task_table(tasks_sorted, settings, betas)

    for i, tk in enumerate(tasks_sorted):
        tid = i + 1
        print(_TASK_HEAD_TITLE_FS.format(tid))
        # PART 1: Utility
        print(_TASK_ANALYSIS_TITLE1_FS)
        t_util = util(tasks_sorted, i, settings, beta=betas[i])
        util_b = util_bound(tid)
        print(_PART1_UTILITY_CPR_FS.format(t_util, "<=" if t_util <= util_b else ">", util_b))
        if t_util <= util_b:
            print(_PART1_UTILITY_PASS)
        else:
            print(_PART1_UTILITY_FAIL)
        # PART 2: Real analysis
        print(_TASK_ANALYSIS_TITLE2_FS)
        r_prev = -1
        valid_stat = True
        r_curr = tasks_sorted[i]["C"] + cost_ctsw * 2 + betas[i]
        print(_PART2_R0_FS.format(r_curr))
        r_iter = 0
        delay_thres = tasks_sorted[i]["D" if "D" in settings and "D" in tasks_sorted[i] else "T"]
        while not math.isclose(r_curr, r_prev):
            r_prev, r_curr = r_curr, 0
            r_iter += 1
            print(_PART2_IT_FS.format(r_iter))
            r_curr = tasks_sorted[i]["C"] + cost_ctsw * 2 + betas[i]
            print(_PART2_BS_FS.format(tid, r_curr))
            for j in range(0, i):
                d_tid = j + 1
                delay = math.ceil(r_prev / tasks_sorted[j]["T"]) * (tasks_sorted[j]["C"] + cost_ctsw * 2)
                print(_PART2_DL_FS.format(d_tid, delay))
                r_curr += delay
            print(_PART2_RN_FS.format(r_iter, r_curr))
            if (not math.isclose(r_curr, delay_thres)) and r_curr > delay_thres:
                valid_stat = False
                break
        print(_PART2_PASS if valid_stat else _PART2_FAIL)
