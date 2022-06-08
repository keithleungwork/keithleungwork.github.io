# Setup VSCode remote development

# Background

A lot of reasons can lead us to use EC2 as development environment.
We can use VSCode directly to manipulate the environment, git, code...etc on the EC2 remotely.

REF:

- [https://code.visualstudio.com/docs/remote/ssh-tutorial](https://code.visualstudio.com/docs/remote/ssh-tutorial)
- [https://guyernest.medium.com/connecting-vs-code-to-develop-machine-learning-models-in-the-aws-cloud-aa1ebd16f890](https://guyernest.medium.com/connecting-vs-code-to-develop-machine-learning-models-in-the-aws-cloud-aa1ebd16f890)

## Prerequisite

Assume you already setup ssh config like this:

```python
Host xxxx_jumpserver
	Hostname <jump ip...>
	User keith
	IdentityFile /Users/xxxxx/.ssh/id_rsa
	Port 23422
	ForwardAgent Yes
	PubKeyAuthentication Yes
	UseKeychain Yes
Host ec2sandbox
	HostName <ec2 host / ip>
	User keith
	ProxyCommand ssh -W %h:%p xxxx_jumpserver
	# Only needed if you want to do port forwarding (tunnel)
	LocalForward 8888 localhost:8888
```

# Procedure

1. Open the `.ssh/config`
    - *Optional* You can install this [vscode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh-edit) to edit ssh config more easier
2. And added port forwarding, such as
    
    ```python
    Host ec2sandbox
      ....
      LocalForward 8888 localhost:8888
      ....
    ```
    
3. Install below official VSCode extension
    - [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
4. You will see a button in bottom left after installing the extension
5. Click "Connect to Host..."
6. The ssh config host which you setup before should appear here as an option. Just choose it and everything would be setup automatically.
7. It is done now! You can click **File -> open folder** to select a remote folder to open as workspace. Also the terminal here would be inside the EC2 already.