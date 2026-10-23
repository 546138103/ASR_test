import unittest
import numpy as np
from realtime_asr import Segmenter


class SegmenterTests(unittest.TestCase):
    def setUp(self):
        self.v = np.full(1600, .1, dtype=np.float32)
        self.z = np.zeros(1600, dtype=np.float32)

    def make(self, **kwargs):
        return Segmenter(sample_rate=16000, threshold=.01, silence_seconds=.3,
                         max_seconds=kwargs.get('max_seconds', 2), min_seconds=.2,
                         preroll_seconds=.2)

    def test_silence_produces_no_segment(self):
        s = self.make()
        for _ in range(50):
            self.assertIsNone(s.feed(self.z))
        self.assertIsNone(s.flush())

    def test_silence_finalizes_and_preserves_preroll(self):
        s = self.make()
        s.feed(self.z)
        s.feed(self.z)
        s.feed(self.v)
        s.feed(self.v)
        self.assertIsNone(s.feed(self.z))
        self.assertIsNone(s.feed(self.z))
        out = s.feed(self.z)
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out[:3200], np.zeros(3200))
        self.assertAlmostEqual(float(out[3200]), .1, places=5)
        self.assertIsNone(s.flush())

    def test_continuous_speech_is_bounded(self):
        s = self.make(max_seconds=.5)
        results = [s.feed(self.v) for _ in range(10)]
        chunks = [r for r in results if r is not None]
        self.assertEqual([len(r) for r in chunks], [8000, 8000])

    def test_flush_preserves_last_phrase_once(self):
        s = self.make()
        s.feed(self.v)
        s.feed(self.v)
        self.assertEqual(len(s.flush()), 3200)
        self.assertIsNone(s.flush())

    def test_short_noise_is_rejected(self):
        s = self.make()
        s.feed(self.v)
        for _ in range(3):
            self.assertIsNone(s.feed(self.z))
        self.assertIsNone(s.flush())

class MicrophoneShutdownTests(unittest.TestCase):
    def run_case(self, interrupted=False, overflows=False):
        import contextlib
        import io
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import realtime_asr as asr

        clock = [0.]
        callback_ref = []
        calls = []
        v = np.full((1600, 1), .1, dtype=np.float32)
        z = np.zeros((1600, 1), dtype=np.float32)

        @contextlib.contextmanager
        def stream(**kwargs):
            cb = kwargs['callback']
            callback_ref.append(cb)
            for frame in [v, v] + [z] * 7:
                cb(frame, len(frame), None, False)
            yield

        fake = SimpleNamespace(check_input_settings=lambda **kw: None,
                               query_devices=lambda *a: {'name': 'test'}, InputStream=stream)
        def decode(model, audio, language):
            calls.append(audio.copy())
            if interrupted and len(calls) == 1:
                raise KeyboardInterrupt
            if overflows:
                callback_ref[0](z, len(z), None, True)
                clock[0] = 10.
            return '尾句'

        args = SimpleNamespace(device=None, threshold=.008, silence=.7, max_seconds=15.,
                               partial_interval=0, duration=5., language='auto')
        output = io.StringIO()
        with patch.dict(sys.modules, {'sounddevice': fake}), \
             patch.object(asr, 'recognize', side_effect=decode), \
             patch.object(asr.time, 'monotonic', side_effect=lambda: clock[0]), \
             contextlib.redirect_stdout(output):
            asr.run_microphone(object(), args)
        return calls, output.getvalue()

    def test_ctrl_c_retries_interrupted_final_utterance(self):
        calls, output = self.run_case(interrupted=True)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], calls[1])
        self.assertIn('[确认] 尾句', output)

    def test_overflow_during_last_inference_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, '溢出|丢帧'):
            self.run_case(overflows=True)


class RecognitionTextTests(unittest.TestCase):
    def decode(self, raw):
        from realtime_asr import recognize
        class Model:
            def generate(self, **kwargs):
                return [{'text': raw}]
        return recognize(Model(), np.zeros(3200, dtype=np.float32), 'auto')

    def test_punctuation_only_noise_is_hidden(self):
        self.assertEqual(self.decode('<|nospeech|><|NEUTRAL|><|Speech|><|withitn|>.'), '')

    def test_transcription_tags_removed_and_words_preserved(self):
        self.assertEqual(self.decode('<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好，Hello 123。'),
                         '你好，Hello 123。')


if __name__ == '__main__':
    unittest.main()
