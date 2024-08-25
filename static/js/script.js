// Function to send the message
function sendMessage() {
    var user_input = document.getElementById('input').value;
    
    if (user_input.trim() !== "") {
        displayUserMessage(user_input);
        document.getElementById('input').value = ''; // Clear the input field

        // Fetch chatbot's response
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({question: user_input})
        }).then(response => response.json())
          .then(data => {
              displayBotMessage(data.answer);

              // Check if FRT is recommended
              if (data.frt_recommended === 1) {
                  displayFRTButtons();
              }
          });
    }
}

// Function to display the FRT buttons
function displayFRTButtons() {
    let buttonContainer = document.getElementById("button-container");
    buttonContainer.innerHTML = ''; // Clear previous buttons if any

    let uploadButton = document.createElement("button");
    uploadButton.textContent = "Upload";
    uploadButton.onclick = function() {
        $('#file-input').click();
    };

    let liveFRTButton = document.createElement("button");
    liveFRTButton.textContent = "Live FRT";
    liveFRTButton.onclick = function() {
        $.ajax({
            url: '/live_frt',
            method: 'GET',
            success: function(response) {
                displayBotMessage('Live FRT result: ' + response.result);
            },
            error: function(error) {
                console.error('Error:', error);
            }
        });
    };

    buttonContainer.appendChild(uploadButton);
    buttonContainer.appendChild(liveFRTButton);
}

// Function to display the user's message
function displayUserMessage(message) {
    let chat = document.getElementById("chat");
    let userMessage = document.createElement("div");
    userMessage.classList.add("message");
    userMessage.classList.add("user");
    let userAvatar = document.createElement("div");
    userAvatar.classList.add("avatar");
    let userText = document.createElement("div");
    userText.classList.add("text");
    userText.innerHTML = message;
    userMessage.appendChild(userAvatar);
    userMessage.appendChild(userText);
    chat.appendChild(userMessage);
    chat.scrollTop = chat.scrollHeight;
}

// Function to display the bot's message
function displayBotMessage(message) {
    let chat = document.getElementById("chat");
    let botMessage = document.createElement("div");
    botMessage.classList.add("message");
    botMessage.classList.add("bot");
    let botAvatar = document.createElement("div");
    botAvatar.classList.add("avatar");
    let botText = document.createElement("div");
    botText.classList.add("text");
    botText.innerHTML = message;
    botMessage.appendChild(botAvatar);
    botMessage.appendChild(botText);
    chat.appendChild(botMessage);
    chat.scrollTop = chat.scrollHeight;
}

// Add a click event listener to the button
document.getElementById("button").addEventListener("click", sendMessage);

// Add a keypress event listener to the input field
document.getElementById("input").addEventListener("keypress", function(event) {
  if (event.keyCode === 13) {
      sendMessage();
  }
});

// Handle video file input change event
$('#file-input').change(function() {
    var file = $('#file-input')[0].files[0];
    var formData = new FormData();
    formData.append('video', file);

    $.ajax({
        url: '/upload',
        method: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(response) {
            displayBotMessage('Upload successful! Video result: ' + response.result);
        },
        error: function(error) {
            console.error('Error:', error);
        }
    });
});
// Example function to update UI with data from the Flask API
function updateUI(data) {
    const logsContainer = document.getElementById('logs-container');  // Assuming you have an element with this ID
    
    logsContainer.innerHTML = '';  // Clear previous logs

    data.logs.forEach(log => {
        // Extract values from the log
        const timestamp = log[0];
        const email = log[1];
        const macAddress = log[2];
        const systolic = log[4];
        const diastolic = log[3];

        // Create a new HTML element for each log
        const logElement = document.createElement('div');
        logElement.innerHTML = `
            <p>Timestamp: ${timestamp}</p>
            <p>Email: ${email}</p>
            <p>MAC Address: ${macAddress}</p>
            <p>Systolic: ${systolic}</p>
            <p>Diastolic: ${diastolic}</p>
        `;
        logsContainer.appendChild(logElement);
    });
}

// Fetch logs from Flask API
fetch('/get-bp-logs')
    .then(response => response.json())
    .then(data => updateUI(data))
    .catch(error => console.error('Error fetching logs:', error));

