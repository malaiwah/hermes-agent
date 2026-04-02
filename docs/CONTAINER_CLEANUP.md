# Container Cleanup Options for Hermes Agent

## Current Cleanup Mechanisms

✅ **What EXISTS:**
1. `DockerEnvironment.cleanup()` - Per-container cleanup on session end
2. `cleanup_all_environments()` - Manual cleanup of ALL active environments
3. `_atexit_cleanup()` - Runs when Python process exits gracefully

❌ **What's MISSING:**
1. ❌ Startup cleanup of orphaned containers
2. ❌ Periodic cleanup cron job
3. ❌ Gateway crash recovery cleanup

## Option 1: Quick Manual Cleanup (IMMEDIATE)

```bash
# On oikos (Alpine LXC), remove ALL exited hermes-* containers:
podman ps -a --filter "name=^hermes-" --filter "status=exited" -q | xargs -r podman rm -f

# Remove ALL hermes-* containers (including stopped ones):
podman ps -a --filter "name=^hermes-" -q | xargs -r podman rm -f

# Remove containers older than 1 day:
podman container prune --filter "until=24h" --filter "name=^hermes-"

# View what would be removed (dry-run):
podman ps -a --filter "name=^hermes-" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
```

## Option 2: Add Startup Cleanup to Gateway (PERMANENT FIX)

**File to modify:** `gateway/run.py`

**Add this function** (after imports, before main code):

```python
def _cleanup_orphaned_containers():
    """Clean up exited hermes-* containers from previous runs.
    
    This prevents accumulation of stopped containers when Hermes
    gateway crashes or restarts without proper cleanup.
    """
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Find all hermes-* containers (stopped or running)
        cmd = [
            "podman", "ps", "-a",
            "--filter", "name=^hermes-",
            "--format", "{{.ID}} {{.Status}}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.debug("Could not list containers: %s", result.stderr)
            return
        
        exited_count = 0
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                container_id = parts[0]
                status = parts[1]
                
                # Only clean up exited/failed containers
                if status.lower() in ['exited', 'dead', 'created']:
                    try:
                        subprocess.run(
                            ["podman", "rm", "-f", container_id],
                            capture_output=True,
                            timeout=30
                        )
                        exited_count += 1
                        logger.info("Cleaned up orphaned container: %s (%s)", 
                                  container_id[:12], status)
                    except Exception as e:
                        logger.warning("Failed to remove container %s: %s", 
                                     container_id[:12], e)
        
        if exited_count > 0:
            logger.info("Startup cleanup: removed %d orphaned hermes-* containers", 
                       exited_count)
            
    except FileNotFoundError:
        logger.debug("Podman not found, skipping container cleanup")
    except Exception as e:
        logger.warning("Container cleanup failed: %s", e)
```

**Call this function** in your gateway startup code (e.g., in `run_gateway()` or similar):

```python
# After establishing logging, before starting gateway loop:
_cleanup_orphaned_containers()
```

## Option 3: Create CLI Command (CONVENIENT)

**File to modify:** `hermes_cli/main.py` or similar CLI entry point

**Add command:**

```python
@cli.command()
@click.option('--dry-run', is_flag=True, help='Show what would be removed')
@click.option('--all', 'remove_all', is_flag=True, help='Remove all hermes-* containers')
@click.option('--older-than', default='24h', help='Remove containers older than this')
def container_cleanup(dry_run: bool, remove_all: bool, older_than: str):
    """Clean up orphaned or old Hermes agent containers."""
    import subprocess
    
    cmd = ["podman", "ps", "-a", "--filter", "name=^hermes-"]
    
    if not remove_all:
        cmd.extend(["--filter", f"until={older_than}"])
    
    if dry_run:
        cmd.extend(["--format", "table {{.ID}}\t{{.Names}}\t{{.Status}}"])
    else:
        cmd.extend(["-q"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    containers = [c for c in result.stdout.strip().split('\n') if c]
    
    if dry_run:
        print("Would remove these containers:")
        print(result.stdout)
    else:
        if containers:
            print(f"Removing {len(containers)} container(s)...")
            for container_id in containers:
                subprocess.run(["podman", "rm", "-f", container_id], 
                             capture_output=True)
            print("Cleanup complete!")
        else:
            print("No containers to clean up.")
```

**Usage:**
```bash
hermes container-cleanup              # Remove containers older than 24h
hermes container-cleanup --all        # Remove ALL hermes-* containers
hermes container-cleanup --dry-run    # Preview what would be removed
hermes container-cleanup --older-than=1h  # Remove containers older than 1 hour
```

## Option 4: Add Cron Job (AUTOMATED)

**On oikos (Alpine LXC), add to crontab:**

```bash
# Clean up exited containers every hour
0 * * * * podman ps -a --filter "name=^hermes-" --filter "status=exited" -q | xargs -r podman rm -f

# Or use the Hermes CLI if Option 3 is implemented
0 * * * * hermes container-cleanup --older-than=1h
```

## Recommended Approach

**For IMMEDIATE relief:**
```bash
# Option 1 - Manual cleanup now
podman ps -a --filter "name=^hermes-" --filter "status=exited" -q | xargs -r podman rm -f
```

**For LONG-TERM fix:**
- Implement **Option 2** (startup cleanup for automatic recovery)
- Implement **Option 3** (CLI command for manual control)
- Optionally **Option 4** (cron for scheduled cleanup)

## Why This Happens

Containers accumulate when:
1. ✅ Hermes gateway crashes (no chance to call cleanup)
2. ✅ Oikos host reboots (containers left in "exited" state)
3. ✅ OOM killer terminates containers (force kill, no cleanup)
4. ✅ Podman restart fails (containers stuck in "created" state)
5. ✅ Manual `podman stop` without `rm` (containers accumulate)

The cleanup function only runs on **graceful exits**, so crashes leave orphans behind.
