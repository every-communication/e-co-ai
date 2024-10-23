async function fetchGesture() {
    try {
        const response = await fetch('/gesture_data');
        if (!response.ok) throw new Error('네트워크 오류가 발생했습니다.');
        
        const data = await response.json();
        const gestureElement = document.getElementById('gesture');
        
        if (data.action && data.action !== "no_hand_detected") {
            gestureElement.textContent = data.action;
        } else {
            gestureElement.textContent = "인식 중...";
        }
    } catch (error) {
        console.error('제스처 데이터를 가져오는 중 오류 발생:', error);
        document.getElementById('gesture').textContent = "오류 발생";
    }
}

// 주기적으로 제스처 데이터를 가져오는 함수 호출
setInterval(fetchGesture, 1000);  // 1초마다 제스처 데이터를 업데이트
