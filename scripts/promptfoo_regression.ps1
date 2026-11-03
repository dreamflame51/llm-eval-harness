# Run the promptfoo regression against a cold scripts/serve.py.
# Run:  pwsh scripts/promptfoo_regression.ps1     -> eval/promptfoo/regression.out.json
#       pwsh scripts/promptfoo_regression.ps1 --filter-first-n 2    (smoke test)
#
# promptfoo is pinned and fetched by npx, not installed. Telemetry, update
# checks and result sharing are off: the questions and answers stay here.

param([Parameter(ValueFromRemainingArguments)] [string[]] $Extra = @())

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
$env:PROMPTFOO_DISABLE_UPDATE = "1"
$env:PROMPTFOO_DISABLE_SHARING = "1"
$env:PROMPTFOO_PYTHON = (Resolve-Path $python).Path

# Long runs: keep the machine awake for as long as this process lives.
Add-Type -Namespace Power -Name State -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Power.State]::SetThreadExecutionState([uint32]2147483649) | Out-Null

& $python scripts/promptfoo_tests.py
& $python -c "from llm_eval_harness import pipeline; pipeline.unload_models()"
if ($LASTEXITCODE) { throw "could not reach Ollama - start it with: ollama serve" }

$server = Start-Process $python -ArgumentList "scripts/serve.py", "--no-open" -PassThru -NoNewWindow
try {
    foreach ($i in 1..60) {
        try { Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing | Out-Null; break } catch { Start-Sleep 2 }
    }
    npx --yes promptfoo@0.124.1 eval -c eval/promptfoo/regression.yaml --max-concurrency 1 --no-cache `
        -o eval/promptfoo/regression.out.json @Extra
} finally {
    Stop-Process -Id $server.Id
}
