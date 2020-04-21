# Start Golang in Windows 10
## Install Go 
1. [Download Golang](https://golang.org/dl/)

2. Install Golang, path environment is set automatically in Windows 10.

3. Check whether Golang is installed correctly by typing `go version`, `go env` in command line

4. The defalut `GOPATH`(workspace) is `C:\Users\yourusername\go`, if not exist create it.
Then create floder in it, for example:`$ mkdir src/golang-book`.

## Two popular IDE:
### VS Code:
1. VS Code is free. [Download VS Code](https://code.visualstudio.com/) and install.  

2. Open vscode, type "ctrl + shif + X", search Go and install.

3.  Settings can be edit by `File` -> `Preferences` -> `Settings` -> `Extensions` -> `Go` -> `Edit in Settings.json`
    Setting `go.inferGopath` overrides the value set in `go.gopath` setting. If `go.inferGopath` is set to true, the extension will try to infer the GOPATH from the path of the workspace i.e. the directory opened in vscode.

    Reference: [GOPATH in the VS Code Go extension](https://github.com/Microsoft/vscode-go/wiki/GOPATH-in-the-VS-Code-Go-extension)

4. Add `.` to enviroment variable `PATH` allow you to run exe without starting with `./`.

### Goland:
    Goland is an IDE from JetBrains (free 30-day trial).
    [Download Goland](https://www.jetbrains.com/go/) 
    

## Hello World in Golang 
```
package main

import "fmt"

func main() {
	fmt.Println("Hello, World")
}
```
### Two way to run the program:

 `$ go run filename.go` to build and execute.

 `$ go build filename.go` to build then `$ filename` to execute.