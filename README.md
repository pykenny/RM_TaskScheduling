# RM_TaskScheduling
Analyze schedulability under rate-monotonic(RM) scheduling. Life is so hard.

## Some examples for usage
```python
# First load the whole code. Module packing or file input will be provided in
# the future, probably.
exec(open("./RMScheduling.py", "r").read())
```

```python
# T: Task period, C: Longest execution time, D: Deadline
tasks = [{'T': 9, 'C': 1.6}, {'T': 8, 'C': 1.6, 'D': 7}, {'T': 2, 'C': 0.6}]
settings = {'CtSw': 0.2, 'D': True} # context switch 0.2, deadline considered
util_analysis(tasks, settings)
```

```python
# B: Blocking
tasks = [{'T': 59, 'C': 26, 'B': 0, 'D': 59}, {'T': 60, 'C': 10, 'B': 2, 'D': 50}, \
         {'T': 155, 'C': 25, 'B': 5, 'D': 135}, {'T': 210, 'C': 15, 'B': 4, 'D': 180}]
settings = {'CtSw': 0.2, 'D': True, 'B': True} # Plus blocking this time
util_analysis(tasks, settings)
```

```python
# Critical resource request
tasks = [{'T': 13, 'C': 3, 'Res': {'R2': 1}}, {'T': 16, 'C': 3, 'Res': {'R1': 1}}, \
         {'T': 21, 'C': 3}, {'T': 31, 'C': 7, 'Res': {'R2': 3, 'R1': 1}}, {'T': 61, 'C': 6, 'Res': {'R1': 4}}]
settings_1 = {'Plcy': 'PIP'} # Only consider critical resource in PIP policy
util_analysis(tasks, settings_1)
settings_2 = {'D': True, 'CtSw': 0.5, "Plcy": "SRP"} # A bit more complicated case
util_analysis(tasks, settings_2)
```
