import csv,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
from experiments.eswa_20261019 import finalize_delivery as module

def test_archive_roundtrip_deduplicates_and_excludes_credentials(tmp_path,monkeypatch):
    main=tmp_path/'results';main.mkdir();(main/'a.txt').write_bytes(b'actual evidence')
    (main/'b.txt').write_bytes(b'actual evidence');(main/'.env.formal').write_bytes(b'not for archive')
    (main/'LIVE_STATUS.json').write_text('{}');(main/'MONITOR_SNAPSHOT.json').write_text('{}')
    output=main/'delivery/v1';output.mkdir(parents=True);(output/'report.md').write_text('honest report')
    monkeypatch.setattr(module,'MAIN',main);module.package(output)
    bundle=output.parent/'v1-evidence.zip';receipt=json.loads((output/'PACKAGE_VERIFICATION.json').read_text())
    assert receipt['logical_files']==3 and receipt['unique_objects']==2
    assert receipt['bundle_sha256']==hashlib.sha256(bundle.read_bytes()).hexdigest()
    with zipfile.ZipFile(bundle) as archive:
        rows=list(csv.DictReader(io.TextIOWrapper(archive.open('FILE_MANIFEST.csv'))))
        assert not any('.env' in row['path'] or 'LIVE_STATUS' in row['path'] for row in rows)
    restored=tmp_path/'restored'
    subprocess.run([sys.executable,str(output/'RESTORE.py'),str(bundle),str(restored)],check=True,capture_output=True)
    assert (restored/'raw/a.txt').read_bytes()==b'actual evidence'
    assert (restored/'raw/b.txt').read_bytes()==b'actual evidence'
    assert (restored/'report/report.md').read_text()=='honest report'
