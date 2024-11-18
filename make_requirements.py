#!/usr/bin/env python3

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import re

def parse_requirement_line(line):
    """Parse a requirement line, handling editable installs and comments."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    try:
        # Parse requirement and return package name
        return Requirement(line).name.lower()
    except:
        return None

def clean_requirements(req_path, cluster_req_path, output_path):
    """Generate clean requirements file excluding cluster packages."""
    
    # Read cluster requirements
    with open(cluster_req_path) as f:
        cluster_packages = {
            parse_requirement_line(line) 
            for line in f 
            if parse_requirement_line(line)
        }
    
    # Read and filter requirements
    with open(req_path) as f, open(output_path, 'w') as out:
        for line in f:
            pkg_name = parse_requirement_line(line)
            
            # Keep line if it's a comment, empty line, or package not in cluster
            if not pkg_name or pkg_name not in cluster_packages:
                out.write(line)

if __name__ == '__main__':
    clean_requirements(
        'requirements.txt',
        'cluster_requirements.txt',
        'clean_requirements.txt'
    )