# Implementation of the Jaccard Timespan Event Score (JTES)

This is a reference implementation of the Jaccard Timespan Event Score (JTES) described [here](https://doi.org/10.1038/s41597-021-00963-2).

Jaccard-Timespan-Event-Score (JTES) is a timespan score based on the Jaccard index also known as Intersection over Union (IoU).

This index is designed to score events that are defined by a timespan defined by two timestamps `(t0, t1)`.

Score Goals:
  * Score is in range `[0-1]`, where 0 is lowest and 1 best.
  * False-Positives and False-Negatives are equally bad
  * If a true event spans over multiple predicted events, the result is averaged
  * If the predicted event spans over multiple true events, the result is accounted accordingly
  * If `len(y_true) == 0 and len(y_pred) > 0`, `score = 0`
  * If `len(y_true) > 0 and len(y_pred) == 0`, `score = 0`
  * If `ordered(y_true) == ordered(y_pred)`, `score = 1`
  * Overlap in `y_true` is not allowed

Jaccard index in general is defined as |A ∩ B| / |A ∪ B|.

## Install
The jtes package is available on [pypi](https://pypi.org/project/jtes/)

```
pip install jtes
```

### Install from source (alternative)
```
python setup.py install
```

## Usage
The `jaccard_timespan_event_score` function expects two [numpy array](https://numpy.org/) `y_true` and `y_pred`.
The events are defined as a [np.datetime64](https://numpy.org/doc/stable/reference/arrays.scalars.html?highlight=datetime64#numpy.datetime64) pairs.
```python
import numpy as np
from jtes import jaccard_timespan_event_score

y_true = np.array([
    (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
    (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00'))
])

y_pred = np.array([
    (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
    (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T05:00:00')),
])

# Returns 0.75
jaccard_timespan_event_score(y_true, y_pred)
```

## License
MIT licensed as found in the [LICENSE](LICENSE) file.
