# Holds ES_CONTINUOUS | ES_SYSTEM_REQUIRED so the machine won't idle-sleep.
# Auto-reverts when this process is killed. No power-plan changes.
$sig = @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);
"@
$api = Add-Type -MemberDefinition $sig -Name Pw -Namespace W -PassThru
while ($true) {
    $null = $api::SetThreadExecutionState(2147483649)  # 0x80000001
    Start-Sleep -Seconds 120
}
