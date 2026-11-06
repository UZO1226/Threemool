"""중간 선택 모델을 마지막 모델로 잘못 평가하지 않는지 확인한다."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import checkpoint_path


class CheckpointTests(unittest.TestCase):
    def test_final_requires_explicit_file_or_last_epoch_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            best = run / 'model/epoch_25'
            best.mkdir(parents=True)
            (best / 'model.pt').touch()
            with self.assertRaises(FileNotFoundError):
                checkpoint_path(run, {'best_epoch': 25}, 'final')
            log = run / 'wandb/offline-run-test/files/output.log'
            log.parent.mkdir(parents=True)
            log.write_text('Epoch #25 ================\n')
            self.assertEqual(checkpoint_path(run, {'best_epoch': 25}, 'final'), best)
            log.write_text('Epoch #25 ================\nEpoch #26 ================\n')
            with self.assertRaises(FileNotFoundError):
                checkpoint_path(run, {'best_epoch': 25}, 'final')
            final = run / 'model_final'
            final.mkdir()
            (final / 'model.pt').touch()
            self.assertEqual(checkpoint_path(run, {'best_epoch': 25}, 'final'), final)


if __name__ == '__main__':
    unittest.main()
