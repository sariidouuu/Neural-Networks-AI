const chatArea    = document.getElementById('chat-area');
const promptInput = document.getElementById('prompt-input');

/* ────── SEND THE PROMPT ────── */
// We use async so the chat will be able to wait for the answer
async function handleSend() {
    const text = promptInput.value.trim();
    //trim: removes whitespace from both ends of a string. So if the user types only spaces, it will be considered empty and won't be sent as a prompt.
    if (!text) return;

    addMessage(text, 'prompt');

    promptInput.value = '';
    promptInput.style.height = 'auto';

    // Small delay to simulate AI "thinking"
    /*setTimeout(() => {
        const response = getAIResponse(text);
        addMessage(response, 'answer');
    }, 400);*/
    // Using await the code awaits the answer from the server
    const response = await getAIResponse(text);
    addMessage(response, 'answer');
}


/* ────── ADD A MESSAGE BUBBLE TO THE CHAT ────── */
function addMessage(text, type) {

    //create a new div element "row"
    const row = document.createElement('div');
    row.classList.add('message-row', type === 'prompt' ? 'prompt-row' : 'answer-row');
    /* if type == prompt then add a class prompt-row, else add a class answer-row */ 

    //create a new div element "bubble"
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', type === 'prompt' ? 'prompt-bubble' : 'answer-bubble');
    /* if type == prompt then add a class prompt-bubble, else add a class answer-bubble */

    //create a new div element "label" for the bubble 
    const label = document.createElement('div');
    label.classList.add('bubble-label');
    label.textContent = type === 'prompt' ? 'You' : 'AI';

    //create a new div element "content" for the text of the bubble
    const content = document.createElement('div');
    content.textContent = text;

    //the text content is the entire bbble, and we create a label for the bubble to show who is speaking (You or AI)
    //and a content area for the text of the bubble.

    //We append the label and content to the bubble, then append the bubble to the row, and finally append the row to the chat area
    bubble.appendChild(label);
    bubble.appendChild(content);
    row.appendChild(bubble);
    chatArea.appendChild(row);

    //The way we append the messages ensures that they are added in the correct order, and the chat area will grow as new messages are added.
    //We use appendChild because it adds in order the elements

    // Scroll to the latest message
    chatArea.scrollTop = chatArea.scrollHeight;
}


/* ────── KEYBOARD EVENTS ────── */
// Enter = send | Shift+Enter = new line
promptInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        //with shift pressed we add a new line, without shift we send the prompt
        e.preventDefault();
        handleSend();
    }
});

// Auto-grow textarea as user types
promptInput.addEventListener('input', function () {
    //1. The event 'input' activates everytime the user types, deletes, or pastes text into the textarea
    this.style.height = 'auto';
    //2. We set the height to 'auto' to reset it, so that if the user deletes text, the textarea can shrink back down in real time
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
    //3. We then set the height to the scrollHeight (the total height of the content), but we cap it at 140px to prevent it from growing too large and taking up too much space in the UI.
});


/* ────── BUTTON FOR SENDING THE PROMPT ────── */
const sendBtn = document.getElementById('send-btn');
sendBtn.addEventListener('click', handleSend);


/* ────── MODEL SELECTOR ────── */
const modelBtn      = document.getElementById('model-btn');
const modelDropdown = document.getElementById('model-dropdown');
const modelOptions  = document.querySelectorAll('.model-option'); //the dot in .model-option indicates class

// Load the saved selection from local storage (basically the selection we already made or the default: 1 -> || 1)
let selectedModel = localStorage.getItem('selectedModel') || '1'; 
updateModelSelection();

modelBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    //The click toogle the dropdown menu, but if we click outside the dropdown, we want to close it. 
    //So we stop the click event from propagating (διάδοση) to the document when we click on the model button.
    modelDropdown.classList.toggle('open');
});

// Close the dropdown if we click elsewhere on the document
document.addEventListener('click', function () {
    modelDropdown.classList.remove('open');
    //If the 'open' class does not exists: Add it to the dropdown model menu
    //If it already exists: Remove it, so the menu closes
});

/*modelOptions.forEach(function (option) {
    option.addEventListener('click', function () {
        selectedModel = this.getAttribute('data-model'); // data-model is the attribute we set in the HTML for each option
        localStorage.setItem('selectedModel', selectedModel);
        updateModelSelection();
        //When we choose the model, we save it to the local storage, and then we close the menuby removing 'open'
        modelDropdown.classList.remove('open');
    });
});*/

// This function finds out the option the user selected and 
// When the user changes the Option, their previous chat is cleared.
modelOptions.forEach(function (option) {
    option.addEventListener('click', function (e) {
        e.stopPropagation();

        const newModel = this.getAttribute('data-model');

        if (newModel === selectedModel) {
            modelDropdown.classList.remove('open');
            return;
        }

        selectedModel = newModel;
        localStorage.setItem('selectedModel', selectedModel);
        updateModelSelection();
        clearChat();
        modelDropdown.classList.remove('open');
    });
});

function clearChat() {
    chatArea.innerHTML = '';
}

//This function updates the visual selection of the model options in the dropdown menu
function updateModelSelection() {
    //Because we use forEach, we loop through all the model options
    modelOptions.forEach(function (option) {
        //We check if the data-model attribute of the option matches the selectedModel variable
        if (option.getAttribute('data-model') === selectedModel) {
            option.classList.add('selected');
            //If it does we add the 'selected' class to that option, otherwise we remove it
        } else {
            option.classList.remove('selected');
        }
    });
}


/* ────── AI RESPONSE BASED ON SELECTED MODEL ────── */
async function getAIResponse(userPrompt) {
    
    if (selectedModel === '1') { 
        //return model1.predict(userPrompt); 
        return "I'm sorry, can't process your request. I'm not trained yet.";
    }
    if (selectedModel === '2') { 
        try {
            // We POST request on our local server (app.py)
            //const response = await fetch('http://127.0.0.1:5000/chat', {
            const response = await fetch('https://neural-networks-ai.onrender.com/chat',{
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json' // We inform our server that we send them the JSON file
                },
                body: JSON.stringify({ message: userPrompt }) // We "package" the question
            });

            // We open the answer package 
            const data = await response.json();

            // If the server sent us back the "reply", we return it to the chat
            if (data.reply) {
                return data.reply;
            } else {
                return "Error: " + (data.error || "SOmething went wrong with API.");
            }
        } catch (error) {
            console.error("Fetch error:", error);
            return "Connection error: Make sure the app.py runs on your terminal!";
        }
    }
    
}