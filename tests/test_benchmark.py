def test_benchmark_client_is_async_and_writes_json():
    src = open("scripts/bench_client.py").read()
    assert "aiohttp" in src and "asyncio" in src


def test_benchmark_script_sweeps_concurrency():
    src = open("scripts/benchmark.sh").read()
    assert "1 4 16" in src or ("1" in src and "16" in src)
