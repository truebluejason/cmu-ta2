#!/usr/bin/env bash

# Make sure our working directory is the jobs directory, relative to the
# scripts directory.
#scriptsdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#cd "$scriptsdir/../jobs"
#
## Confirm
#read -p "Are you sure you want to remove ALL jobs output (rm -rf ${jobsdir}/*)? " -n 1 -r
#echo    # (optional) move to a new line
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
#    rm -rf "${jobsdir}"/*
#    echo "Removed ${jobsdir}/*"
#fi