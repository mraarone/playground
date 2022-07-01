##To connect on another computer, open port 2375 on the server, and use this code in **settings,hjson**.
```
    {
        "remote.extensionKind": {
            "ms-azuretools.vscode-docker": [ "ui" ],
        },
        "docker.host":"tcp://<remote_ip>:2375"
    }
```
