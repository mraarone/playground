[![Open in Remote - Containers](https://img.shields.io/static/v1?label=Remote%20-%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/mraarone/playground)

# Sandbox

## Remote Connections

To connect on another computer, open port 2375 on the server, and use this code in **settings.json**

```json
{
    "remote.extensionKind": {
        "ms-azuretools.vscode-docker": [ "ui" ],
    },
    "docker.host":"tcp://<remote_ip>:2375"
}
```
