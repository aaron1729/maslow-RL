#!/usr/bin/env python3
"""
Script to deploy and run training on Lambda Labs GPU instance.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

LAMBDA_API_KEY = os.getenv('LAMBDA_API_KEY')
BASE_URL = 'https://cloud.lambdalabs.com/api/v1'
HEADERS = {'Authorization': f'Bearer {LAMBDA_API_KEY}'}


def check_ssh_key():
    """Check if SSH key exists, create if not."""
    ssh_dir = Path.home() / '.ssh'
    key_path = ssh_dir / 'id_rsa'
    pub_key_path = ssh_dir / 'id_rsa.pub'

    if not pub_key_path.exists():
        print("No SSH key found. Generating one...")
        subprocess.run(['ssh-keygen', '-t', 'rsa', '-b', '4096', '-f', str(key_path), '-N', ''])

    with open(pub_key_path) as f:
        return f.read().strip()


def add_ssh_key_to_lambda(ssh_public_key):
    """Add SSH key to Lambda Labs account."""
    # Check existing keys
    response = requests.get(f'{BASE_URL}/ssh-keys', headers=HEADERS)

    if response.status_code == 200:
        existing_keys = response.json()['data']
        for key in existing_keys:
            if key['public_key'].strip() == ssh_public_key.strip():
                print(f"SSH key already registered: {key['name']}")
                return key['name']  # Return name, not ID

    # Add new key
    key_name = f"maslow-rl-{int(time.time())}"
    data = {
        'name': key_name,
        'public_key': ssh_public_key
    }

    response = requests.post(f'{BASE_URL}/ssh-keys', headers=HEADERS, json=data)

    if response.status_code == 200:
        key_id = response.json()['data']['id']
        print(f"SSH key added: {key_name} (ID: {key_id})")
        return key_name  # Return name, not ID
    else:
        print(f"Error adding SSH key: {response.status_code}")
        print(response.text)
        sys.exit(1)


def launch_instance(instance_type='gpu_1x_a10', region='us-east-1'):
    """Launch a Lambda Labs instance."""
    print(f"\nLaunching {instance_type} in {region}...")

    # Get SSH key
    ssh_public_key = check_ssh_key()
    ssh_key_name = add_ssh_key_to_lambda(ssh_public_key)

    # Launch instance
    data = {
        'region_name': region,
        'instance_type_name': instance_type,
        'ssh_key_names': [ssh_key_name],
        'quantity': 1
    }

    print(f"Request data: {json.dumps(data, indent=2)}")
    response = requests.post(f'{BASE_URL}/instance-operations/launch', headers=HEADERS, json=data)

    if response.status_code == 200:
        result = response.json()
        instance_ids = result['data']['instance_ids']
        print(f"✓ Instance launched: {instance_ids[0]}")
        return instance_ids[0]
    else:
        print(f"\nError launching instance: {response.status_code}")
        print(f"Response: {response.text}")
        try:
            error_data = response.json()
            print(f"\nDetailed error: {json.dumps(error_data, indent=2)}")
        except:
            pass
        sys.exit(1)


def wait_for_instance(instance_id):
    """Wait for instance to be running and get IP."""
    print("\nWaiting for instance to be ready...")

    for i in range(60):  # Wait up to 5 minutes
        response = requests.get(f'{BASE_URL}/instances', headers=HEADERS)

        if response.status_code == 200:
            instances = response.json()['data']
            for inst in instances:
                if inst['id'] == instance_id:
                    status = inst['status']
                    print(f"  Status: {status}")

                    if status == 'active':
                        ip = inst['ip']
                        print(f"✓ Instance ready! IP: {ip}")
                        return ip

        time.sleep(5)

    print("Timeout waiting for instance")
    sys.exit(1)


def upload_code(ip):
    """Upload code to the instance."""
    print("\nUploading code to instance...")

    # Files to upload
    files = [
        'config.json',
        'config_linear.json',
        'data.py',
        'rewards.py',
        'train.py',
        'eval.py',
        'requirements.txt',
        '.env'
    ]

    remote_dir = 'maslow-rl'

    # Create directory on remote
    subprocess.run(['ssh', '-o', 'StrictHostKeyChecking=no', f'ubuntu@{ip}', f'mkdir -p {remote_dir}'])

    # Upload files
    for file in files:
        if Path(file).exists():
            print(f"  Uploading {file}...")
            subprocess.run(['scp', '-o', 'StrictHostKeyChecking=no', file, f'ubuntu@{ip}:{remote_dir}/'])

    print("✓ Code uploaded")


def setup_environment(ip):
    """Set up Python environment on remote instance."""
    print("\nSetting up environment...")

    commands = [
        'cd maslow-rl',
        'python3 -m venv venv',
        'source venv/bin/activate',
        'pip install --upgrade pip',
        'pip install -r requirements.txt'
    ]

    cmd = ' && '.join(commands)
    subprocess.run(['ssh', '-o', 'StrictHostKeyChecking=no', f'ubuntu@{ip}', cmd])

    print("✓ Environment set up")


def run_training(ip, config='config.json', background=False):
    """Run training on remote instance."""
    config_name = Path(config).stem
    print(f"\nStarting training: {config_name}...")

    if background:
        # Run in background with nohup - need to wrap in bash -c
        cmd = f'bash -c "cd maslow-rl && source venv/bin/activate && nohup python train.py --config {config} > {config_name}.log 2>&1 &"'
        subprocess.run(['ssh', '-o', 'StrictHostKeyChecking=no', f'ubuntu@{ip}', cmd])
        print(f"✓ Training started in background (logs: {config_name}.log)")
    else:
        # Run interactively
        cmd = f'cd maslow-rl && source venv/bin/activate && python train.py --config {config}'
        subprocess.run(['ssh', '-t', '-o', 'StrictHostKeyChecking=no', f'ubuntu@{ip}', cmd])


def main():
    if not LAMBDA_API_KEY:
        print("Error: LAMBDA_API_KEY not found in .env")
        sys.exit(1)

    print("=" * 80)
    print("LAMBDA LABS DEPLOYMENT - Maslow RL")
    print("=" * 80)

    # Launch instance
    instance_id = launch_instance()

    # Wait for it to be ready
    ip = wait_for_instance(instance_id)

    # Save instance info
    with open('.lambda_instance.json', 'w') as f:
        json.dump({'instance_id': instance_id, 'ip': ip}, f)

    print(f"\n✓ Instance info saved to .lambda_instance.json")

    # Upload code
    upload_code(ip)

    # Setup environment
    setup_environment(ip)

    print("\n" + "=" * 80)
    print("INSTANCE READY!")
    print("=" * 80)
    print(f"IP: {ip}")
    print(f"SSH: ssh ubuntu@{ip}")
    print(f"Instance ID: {instance_id}")

    print("\nNext steps:")
    print("1. Run gated training: python lambda_deploy.py run-gated")
    print("2. Run linear training: python lambda_deploy.py run-linear")
    print("3. Run both in parallel: python lambda_deploy.py run-both")
    print("4. SSH to instance: ssh ubuntu@{ip}")
    print("5. Terminate when done: python lambda_deploy.py terminate")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        # Load instance info
        if not Path('.lambda_instance.json').exists():
            print("No instance found. Run without arguments to launch one.")
            sys.exit(1)

        with open('.lambda_instance.json') as f:
            info = json.load(f)
            ip = info['ip']
            instance_id = info['instance_id']

        if command == 'run-gated':
            run_training(ip, 'config.json', background=False)
        elif command == 'run-linear':
            run_training(ip, 'config_linear.json', background=False)
        elif command == 'run-both':
            print("Starting both training runs in parallel...")
            run_training(ip, 'config.json', background=True)
            time.sleep(2)
            run_training(ip, 'config_linear.json', background=True)
            print("\n✓ Both runs started!")
            print(f"Monitor: ssh ubuntu@{ip}")
            print("  tail -f maslow-rl/config.log")
            print("  tail -f maslow-rl/config_linear.log")
        elif command == 'terminate':
            print(f"Terminating instance {instance_id}...")
            response = requests.post(
                f'{BASE_URL}/instance-operations/terminate',
                headers=HEADERS,
                json={'instance_ids': [instance_id]}
            )
            if response.status_code == 200:
                print("✓ Instance terminated")
                Path('.lambda_instance.json').unlink()
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        else:
            print(f"Unknown command: {command}")
    else:
        main()
