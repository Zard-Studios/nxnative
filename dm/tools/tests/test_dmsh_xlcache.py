"""The translation cache (dm/gfx/tools/dmsh_xlcache.py): a job's translator files come back byte for byte, only for the same job and the same translator binary."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/gfx/tools'))
sys.path.insert(0, str(REPO / 'dm/tools'))
import dmsh_xlcache  # noqa: E402
import shader_cache  # noqa: E402


class XlCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.xlate = self.root / 'dmsh_xlate'; self.xlate.write_bytes(b'translator one')
        self.store = shader_cache.Store(self.root / 'cas.db')
        self.addCleanup(self.store.close); self.addCleanup(self.tmp.cleanup)

    def folder(self, name):
        d = self.root / name; d.mkdir(); return d

    def job(self, folder, code=b'\x01\x02', **extra):
        p = folder / 'k.bin'; p.write_bytes(code)
        return {'key': 'k', 'stage': 'ps', 'code': str(p), 'c1': '00', 'attr': {}, **extra}

    def translated(self, folder):   # what dmsh_xlate writes for a job
        (folder / 'k.spv').write_bytes(b'\x03\x02\x23\x07spirv')
        (folder / 'k.res.json').write_text('{"notes": ["translator: a message"]}')

    def test_round_trip(self):
        a = self.folder('a'); j = self.job(a); self.translated(a)
        c = dmsh_xlcache.Cache(self.store, str(self.xlate))
        self.assertFalse(c.take(j, a, 'k'))
        c.keep(j, a, 'k', False)
        b = self.folder('b'); jb = self.job(b)   # the same job from another working folder
        self.assertTrue(c.take(jb, b, 'k'))
        for f in ('k.spv', 'k.res.json'): self.assertEqual((b / f).read_bytes(), (a / f).read_bytes())
        self.assertFalse((b / 'k.reader.spv').exists())
        self.assertEqual(c.stats, {'translated': 1, 'reused': 1})

    def test_another_job_or_binary_misses(self):
        a = self.folder('a'); j = self.job(a); self.translated(a)
        c = dmsh_xlcache.Cache(self.store, str(self.xlate)); c.keep(j, a, 'k', False)
        b = self.folder('b')
        self.assertFalse(c.take(self.job(b, code=b'\x01\x03'), b, 'k'))   # other code
        self.assertFalse(c.take(self.job(b, attr={'0': 'uint'}), b, 'k'))   # other fields: the vertex shader's integer-input retranslation
        self.xlate.write_bytes(b'translator two')
        self.assertFalse(dmsh_xlcache.Cache(self.store, str(self.xlate)).take(self.job(b), b, 'k'))

    def test_failed_translation_is_not_kept(self):
        a = self.folder('a'); j = self.job(a); self.translated(a)
        c = dmsh_xlcache.Cache(self.store, str(self.xlate)); c.keep(j, a, 'k', True)
        (a / 'k.spv').unlink(); c.keep(j, a, 'k', False)   # no SPIR-V written
        self.assertEqual(self.store.names('xlate/'), set())

    def test_off(self):
        a = self.folder('a'); j = self.job(a); self.translated(a)
        with mock.patch.dict(os.environ, {'DMSH_XL_CACHE': '0'}):
            c = dmsh_xlcache.Cache(self.store, str(self.xlate))
        c.keep(j, a, 'k', False)
        self.assertFalse(c.take(j, a, 'k'))
        self.assertEqual(self.store.names('xlate/'), set())

    def test_damaged_entry_is_a_miss(self):
        a = self.folder('a'); j = self.job(a)
        c = dmsh_xlcache.Cache(self.store, str(self.xlate))
        self.store.put(c.prefix + c.digest(j), b'.spv\x00\xff\xff\xff\xff\xff\xff\xff\x7fshort')
        self.assertFalse(c.take(j, a, 'k'))
        self.assertFalse((a / 'k.spv').exists())

    def test_prune_keeps_the_current_binary(self):
        a = self.folder('a'); j = self.job(a); self.translated(a)
        old = dmsh_xlcache.Cache(self.store, str(self.xlate)); old.keep(j, a, 'k', False)
        self.xlate.write_bytes(b'translator two')
        new = dmsh_xlcache.Cache(self.store, str(self.xlate)); new.keep(j, a, 'k', False)
        new.prune()
        self.assertEqual({n.split('/')[1] for n in self.store.names('xlate/')}, {new.fp})


if __name__ == '__main__':
    unittest.main()
