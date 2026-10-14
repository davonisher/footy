import Cocoa
import WebKit

/// Transparent strip so the window can be dragged (WKWebView ignores -webkit-app-region).
final class WindowDragView: NSView {
    override var mouseDownCanMoveWindow: Bool { true }
    override func draw(_ dirtyRect: NSRect) {}
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    var serverProcess: Process?
    let port = 5055
    let root = NSString(string: "~/Code/batch-transcriber").expandingTildeInPath

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        startServerIfNeeded()
        buildWindow()
        waitForServerAndLoad()
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopServer()
    }

    private func buildWindow() {
        let rect = NSRect(x: 0, y: 0, width: 920, height: 640)
        window = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Footy"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.minSize = NSSize(width: 560, height: 420)
        window.isOpaque = true
        window.isReleasedWhenClosed = false
        window.isMovable = true
        window.isMovableByWindowBackground = true
        window.setContentSize(rect.size)
        window.collectionBehavior.insert(.fullScreenPrimary)
        window.center()
        window.backgroundColor = NSColor(calibratedRed: 0.961, green: 0.961, blue: 0.969, alpha: 1)

        let content = NSView(frame: rect)
        content.wantsLayer = true
        content.layer?.backgroundColor = window.backgroundColor.cgColor

        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        webView = WKWebView(frame: content.bounds, configuration: config)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")
        content.addSubview(webView)

        // Drag only over the sidebar/title strip so main UI stays clickable.
        let dragHeight: CGFloat = 52
        let dragWidth: CGFloat = 240
        let drag = WindowDragView(frame: NSRect(
            x: 0,
            y: content.bounds.height - dragHeight,
            width: dragWidth,
            height: dragHeight
        ))
        drag.autoresizingMask = [.minYMargin]
        content.addSubview(drag)

        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(webView)
    }

    private func pythonPath() -> String {
        root + "/.venv/bin/python"
    }

    private func serverIsUp() -> Bool {
        var pointer: UnsafeMutablePointer<addrinfo>?
        var hints = addrinfo()
        hints.ai_family = AF_INET
        hints.ai_socktype = SOCK_STREAM
        let host = "127.0.0.1"
        let portStr = String(port)
        guard getaddrinfo(host, portStr, &hints, &pointer) == 0, let addr = pointer else {
            return false
        }
        defer { freeaddrinfo(pointer) }
        let fd = socket(addr.pointee.ai_family, addr.pointee.ai_socktype, addr.pointee.ai_protocol)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        let result = connect(fd, addr.pointee.ai_addr, addr.pointee.ai_addrlen)
        return result == 0
    }

    private func startServerIfNeeded() {
        if serverIsUp() { return }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/arch")
        process.arguments = ["-arm64", pythonPath(), root + "/app.py"]

        let logPath = root + "/work/server.log"
        try? FileManager.default.createDirectory(
            atPath: root + "/work",
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let logUrl = URL(fileURLWithPath: logPath)
        if let handle = try? FileHandle(forWritingTo: logUrl) {
            process.standardOutput = handle
            process.standardError = handle
        }

        process.environment = ProcessInfo.processInfo.environment
        var env = process.environment ?? [:]
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "")
        process.environment = env

        do {
            try process.run()
            serverProcess = process
        } catch {
            showAlert("Kon de transcriptie-server niet starten:\n\(error.localizedDescription)")
        }
    }

    private func stopServer() {
        if let process = serverProcess, process.isRunning {
            process.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.4) {
                if process.isRunning {
                    process.terminate()
                    kill(process.processIdentifier, SIGKILL)
                }
            }
        }
        serverProcess = nil
    }

    private func waitForServerAndLoad() {
        DispatchQueue.global(qos: .userInitiated).async {
            var ok = false
            for _ in 0..<60 {
                if self.serverIsUp() {
                    ok = true
                    break
                }
                Thread.sleep(forTimeInterval: 0.2)
            }
            DispatchQueue.main.async {
                if ok, let url = URL(string: "http://127.0.0.1:\(self.port)/") {
                    self.webView.load(URLRequest(url: url))
                } else {
                    self.showAlert("Server startte niet. Zie ~/Code/batch-transcriber/work/server.log")
                }
            }
        }
    }

    private func showAlert(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Footy"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.runModal()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
