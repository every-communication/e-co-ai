document.addEventListener('DOMContentLoaded', () => {
    const actionElement = document.getElementById('action');

    // EventSource 객체를 사용하여 서버로부터 실시간 데이터 스트리밍 받음
    const eventSource = new EventSource('/stream-data');

    eventSource.onmessage = function(event) {
        // 서버에서 받은 데이터를 화면에 출력
        actionElement.innerText = event.data;
    };

    eventSource.onerror = function(error) {
        console.error('Error fetching stream data:', error);
    };
});
