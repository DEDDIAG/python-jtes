from unittest import TestCase

from jtes import jaccard_timespan_event_score
import numpy as np


class JTESTest(TestCase):

    def test_both_empty(self):
        self.assertEqual(1, jaccard_timespan_event_score(np.array([]), np.array([])))

    def test_empty_prediction(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:10'))
        ])
        self.assertEqual(0, jaccard_timespan_event_score(y_true, np.array([])))

    def test_empty_true(self):
        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:10'))
        ])
        self.assertEqual(0, jaccard_timespan_event_score(np.array([]), y_pred))

    def test_full_overlap(self):
        y = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:10'))
        ])

        self.assertEqual(1, jaccard_timespan_event_score(y, y))

    def test_single_half_overlap(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00'))
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T05:00:00')),
        ])
        self.assertEqual(0.75, jaccard_timespan_event_score(y_true, y_pred))

    def test_double_half_overlap(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00'))
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T02:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T05:00:00')),
        ])
        self.assertEqual(0.5, jaccard_timespan_event_score(y_true, y_pred))

    def test_false_positive(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
            (np.datetime64('1901-01-01T03:00:00'), np.datetime64('1901-01-01T03:00:05')),
        ])
        self.assertEqual(2 / 3, jaccard_timespan_event_score(y_true, y_pred))

    def test_false_negative(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
            (np.datetime64('1901-01-01T03:00:00'), np.datetime64('1901-01-01T03:00:05')),
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
        ])
        self.assertEqual(2 / 3, jaccard_timespan_event_score(y_true, y_pred))

    def test_duplicates(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),  # dub
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),  # dub
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),  # dub
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),  # dub
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:05')),  # dub
        ])
        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(y_true, y_pred)

    def test_simple_pred_split(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T00:20:00')),
            (np.datetime64('1900-01-01T00:20:00'), np.datetime64('1900-01-01T01:00:00')),
        ])
        self.assertEqual(0.5, jaccard_timespan_event_score(y_true, y_pred))

    def test_pred_split(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00'))
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T00:20:00')),
            (np.datetime64('1900-01-01T00:20:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:10:00')),
            (np.datetime64('1900-01-01T03:10:00'), np.datetime64('1900-01-01T03:45:00')),
            (np.datetime64('1900-01-01T03:45:00'), np.datetime64('1900-01-01T04:00:00'))
        ])
        self.assertEqual(((20/60 + 40/60)/2 + (10/60 + 35/60 + 15/60)/3)/2,
                         jaccard_timespan_event_score(y_true, y_pred))

    def test_pred_overlap(self):
        y_true = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00'))
        ])

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T00:30:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:18:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T04:00:00')),
            (np.datetime64('1900-01-01T03:10:00'), np.datetime64('1900-01-01T03:44:00'))
        ])
        self.assertEqual(((0.5+1)/2 + (18/60+1+34/60)/3)/2, jaccard_timespan_event_score(y_true, y_pred))

        y_pred = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T04:00:00')),
        ])
        self.assertEqual((1/4 + 1/4)/2, jaccard_timespan_event_score(y_true, y_pred))

    def test_t0_lt_t1(self):
        j_wrong = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:10'), np.datetime64('1900-01-01T03:00:00'))  # t0 > t1
        ])
        y_correct = np.array([
            (np.datetime64('1900-01-01T00:00:00'), np.datetime64('1900-01-01T01:00:00')),
            (np.datetime64('1900-01-01T03:00:00'), np.datetime64('1900-01-01T03:00:10'))
        ])

        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(j_wrong, np.array([]))
        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(np.array([]), j_wrong)
        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(j_wrong, j_wrong)
        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(j_wrong, y_correct)
        with self.assertRaises(ValueError):
            jaccard_timespan_event_score(y_correct, j_wrong)
