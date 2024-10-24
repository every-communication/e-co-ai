const { spawn } = require('child_process');
const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// 정적 파일 제공 설정 (public 폴더)
app.use(express.static(path.join(__dirname, 'public')));

// 루트 경로로 접속했을 때 index.html 제공
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Python 스크립트 실행 및 실시간 데이터 스트리밍
app.get('/stream-data', (req, res) => {
    // 헤더 설정 (EventStream 형식으로 전송)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const pythonProcess = spawn('python', ['app.py']); // Python 스크립트 실행

    pythonProcess.stdout.on('data', (data) => {
        // 데이터를 클라이언트에 EventStream 방식으로 전송
        res.write(`data: ${data.toString()}\n\n`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error from Python: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        res.write(`event: end\ndata: Python process exited with code ${code}\n\n`);
        res.end();  // Python 프로세스가 끝나면 응답을 마침
        console.log(`Python process exited with code ${code}`);
    });
});

// 서버 실행
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
