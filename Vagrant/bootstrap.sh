#!/usr/bin/env bash

export DEBIAN_FRONTEND=noninteractive
# My Essential Commands:
apt-get update
apt-get -y upgrade
apt-get -y install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev \
	libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev liblzma-dev \
	libpcre3 libpcre3-dev git mc nano htop curl wget

# Creating the swap space 1GB
echo "Creating 1GB swap space in /swapfile..."
fallocate -l 1G /swapfile
ls -lh /swapfile
# Securing the swapfile
echo "Securing the swapfile..."
chown root:root /swapfile
chmod 0600 /swapfile
ls -lh /swapfile
# Turning the swapfile on
echo "Turning the swapfile on..."
mkswap /swapfile
swapon /swapfile
# Verifying
echo "Verifying..."
swapon -s
grep -i --color swap /proc/meminfo
# Adding swap entry to /etc/fstab
echo "Adding swap entry to /etc/fstab"
echo "/swapfile none swap sw 0 0" >> /etc/fstab
# Verifying
echo "Result: "
cat /etc/fstab
# Done!
echo " ****** We are done !!! ********"
