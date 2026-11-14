#!/usr/bin/env python3
# Translation cache for dmsh_build.py: dmsh_xlate's outputs for a shader (<key>.spv, <key>.res.json, <key>.reader.spv) depend only on its job (dmsh_build.py job_for: the
# code, c1, the facts and the options) and on the translator binary; the translator's environment switches only trace to stderr. So a shader whose job is the same as in an
# earlier build, with the same binary, takes that translation's files from the shader cache file (entries xlate/<binary digest>/<job digest>) instead of running the translator.
# dmsh_build.py's per-shader key (shader_key) covers every chain tool, so a change to one Python rule (dmsh_rewrite.py, dmsh_fetch.py, ...) rebuilt every shader through the
# translator although its outputs had not changed; with this cache such a change costs the rules, spirv-cross and the Metal compiles (themselves reused by MSL, dmsh_mtlcache.py).
# Like dmsh_mtlcache.py this file is not part of the chain version: the files it restores are the translator's, byte for byte. DMSH_XL_CACHE=0 turns it off.
import os, json, struct, hashlib, collections

FORMAT = b'dmsh_xlcache 1'
FILES = ('.spv', '.res.json', '.reader.spv')   # what dmsh_xlate/Program.cs writes for a job


class Cache:
    def __init__(self, store, xlate):
        self.store, self.on = store, os.environ.get('DMSH_XL_CACHE', '1') != '0'
        self.stats = collections.Counter()   # reused, translated: summed from dmsh_build.py's worker processes like its other counters
        with open(xlate, 'rb') as f: self.fp = hashlib.sha256(f.read()).hexdigest()[:24] if self.on else ''
        self.prefix = f'xlate/{self.fp}/'

    @staticmethod
    def digest(job):   # the job as the translator reads it: its fields, with the code's bytes in place of its working path
        with open(job['code'], 'rb') as f: code = f.read()
        fields = json.dumps({k: v for k, v in job.items() if k != 'code'}, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(FORMAT + b'\0' + fields + b'\0' + code).hexdigest()

    @staticmethod
    def pack(folder, key):   # the translation's files as one entry: for each, its suffix, a nul, its length (8 bytes) and its bytes
        out = []
        for sfx in FILES:
            p = os.path.join(folder, key + sfx)
            if os.path.exists(p):
                with open(p, 'rb') as f: data = f.read()
                out.append(sfx.encode() + b'\0' + struct.pack('<Q', len(data)) + data)
        return b''.join(out)

    @staticmethod
    def unpack(data, folder, key):   # the files back into the working folder; False for an entry pack did not write
        files, p = {}, 0
        while p < len(data):
            z = data.find(b'\0', p)
            if z < 0 or z + 9 > len(data): return False
            n = struct.unpack_from('<Q', data, z + 1)[0]
            if z + 9 + n > len(data): return False
            files[data[p:z].decode()] = data[z + 9:z + 9 + n]; p = z + 9 + n
        if '.spv' not in files or not set(files) <= set(FILES): return False
        for sfx, b in files.items():
            with open(os.path.join(folder, key + sfx), 'wb') as f: f.write(b)
        return True

    def take(self, job, folder, key):   # True when the cache had this job's translation, now in folder
        if not self.on: return False
        data = self.store.get(self.prefix + self.digest(job))
        if data and self.unpack(data, folder, key): self.stats['reused'] += 1; return True
        return False

    def keep(self, job, folder, key, failed):   # a translation the translator just made, for later builds (not one it reported an error for)
        if not self.on or failed or not os.path.exists(os.path.join(folder, key + '.spv')): return
        self.store.put(self.prefix + self.digest(job), self.pack(folder, key)); self.stats['translated'] += 1

    def prune(self):   # after a full build: the translations of other translator binaries, which no build of this tree reads
        if not self.on: return
        for fp in {n.split('/')[1] for n in self.store.names('xlate/')} - {self.fp}: self.store.delete_prefix(f'xlate/{fp}/')

    def summary(self):
        return f'translation cache: {self.stats["reused"]} translations reused, {self.stats["translated"]} translated' if self.on else 'translation cache: off (DMSH_XL_CACHE=0)'
