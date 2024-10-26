const video = document.getElementById('webcam');
const actionText = document.getElementById('action');
const socket = io.connect('http://localhost:5000');

// 웹캠 접근 권한 요청 및 스트림 설정
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        setInterval(captureAndSendFrame, 1000);  // 1초마다 프레임 전송
    })
    .catch(error => {
        console.error("웹캠 접근 실패:", error);
    });

// 캡처한 프레임을 서버에 전송하는 함수
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 캡처된 프레임을 Blob 형식으로 변환하여 전송
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            socket.emit('image', { image: reader.result });
        };
        reader.readAsArrayBuffer(blob);
    });
}

// 서버로부터 번역 결과를 받았을 때 화면에 표시
socket.on('response', data => {
    console.log('서버 응답:', data);
    actionText.textContent = data.result ? data.result : "인식 중...";
});
