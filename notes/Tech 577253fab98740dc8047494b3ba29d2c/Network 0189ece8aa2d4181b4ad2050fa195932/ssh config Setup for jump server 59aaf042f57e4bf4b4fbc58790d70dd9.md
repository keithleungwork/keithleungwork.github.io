# ssh/config Setup for jump server

# Guide

## Intro

Assuming your situation is that connecting to a jump server is a must, in order to access EC2.

And now you don't want to connect to jump server manually every time, follow the procedure below to simplify the process.

---

# Setup .ssh/config

## Short Method:

```python
Host xxxx_jumpserver
	Hostname <jump server ip>
	User keith
	IdentityFile /Users/xxxx/.ssh/id_rsa
	Port 23422
	ForwardAgent Yes
	PubKeyAuthentication Yes
	UseKeychain Yes
Host ec2sandbox
	HostName <ec2 ip/hostname>
	User keith
	ProxyCommand ssh -W %h:%p xxxx_jumpserver
	# Only needed if you want to do port forwarding (tunnel)
	LocalForward 8888 localhost:8888
```

## Long method:

Modify the `.ssh/config` as below

```
... some other lines...

Host <jump server ip address here>
        ForwardAgent yes
        IdentityFile <Full path to your key file>
        UseKeychain yes

... some other lines...

```

Replace the `<jump server ip address here>`  as the jump server ip
And replace the <Full path to your key file> as the FULL path to your key file

Then you are done !
Simply type below to connect to ec2

```
ssh  -J <your username>@<jump server ip>:23422 <ec2 username>@<your ec2 host>

```