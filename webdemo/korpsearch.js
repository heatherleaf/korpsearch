
const API_DOMAIN = "http://127.0.0.1:8000/";
const LOCALE = "en-US";

const ELEMS = {
    corpus: null,
    search: null,
    navigate: null,
    error: null,
}

const search_examples = [
    '[pos="ART"] [lemma="big"] [pos="SUBST"]',
    '[lemma="be"] [pos="ART"] [pos="ADJ"] [pos="SUBST"]',
    '[pos="ADJ"] [lemma="cut" pos="VERB"]',
]

window.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    ELEMS.corpus = {
        list: document.getElementById('corpus-list'),
        info: document.getElementById('corpus-info'),
    };
    ELEMS.search = {
        string: document.getElementById('search-string'),
        examples: document.getElementById('search-examples'),
        button: document.getElementById('search-button'),
        info: document.getElementById('search-info'),
        results: document.getElementById('search-results'),
        container: document.getElementById('search-results-container'),
    },
    ELEMS.navigate = {
        first: document.getElementById('navigate-first'),
        prev: document.getElementById('navigate-prev'),
        next: document.getElementById('navigate-next'),
        last: document.getElementById('navigate-last'),
    }
    ELEMS.error = document.getElementById('error');

    call_api('info', {}, (response) => {
        ELEMS.corpus.list.innerHTML = "";
        for (let corpus of response.corpora) {
            let opt = document.createElement('option');
            opt.text = opt.value = corpus;
            ELEMS.corpus.list.add(opt);
        }
        select_corpus();
    })

    let opt = document.createElement('option');
    opt.text = '(try an example search)'
    ELEMS.search.examples.add(opt);
    for (let ex of search_examples) {
        let opt = document.createElement('option');
        opt.text = opt.value = ex;
        ELEMS.search.examples.add(opt);
    }

    ELEMS.corpus.list.addEventListener('change', select_corpus);
    ELEMS.search.examples.addEventListener('change', select_example);
    ELEMS.search.button.addEventListener('click', search_corpus);

    for (let nav in ELEMS.navigate) {
        ELEMS.navigate[nav].addEventListener('click', () => navigate(nav));
    }

    ELEMS.search.string.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault(); // Cancel the default action, if needed
            ELEMS.search.button.click();
        }
    });
}


function clear_output() {
    ELEMS.corpus.info.innerHTML = ELEMS.search.info.innerHTML = ELEMS.search.results.innerHTML = ELEMS.error.innerHTML = '';
}


function select_corpus() {
    clear_output();
    let corpus = ELEMS.corpus.list.value;
    call_api('corpus_info', {corpus: corpus}, (response) => {
        let html = "";
        for (const corpus of Object.values(response.corpora)) {
            html += `<p>
                <strong>${corpus.info.Name}</strong>:
                ${corpus.info.Size.toLocaleString(LOCALE)} tokens;
                ${corpus.info.Sentences.toLocaleString(LOCALE)} sentences;
                ${corpus.info.Indexes.length.toLocaleString(LOCALE)} compiled indexes;
                ${corpus.attrs.p.length.toLocaleString(LOCALE)} features (<em>${corpus.attrs.p.join('</em>, <em>')}</em>).
            </p>`;
        }
        ELEMS.corpus.info.innerHTML = html;
    });
}


function select_example() {
    let examples = ELEMS.search.examples;
    if (examples.value) {
        ELEMS.search.string.value = examples.value;
        ELEMS.search.string.focus();
    }
    examples.selectedIndex = 0;
}


const NUM_HITS = 20;

function search_corpus() {
    let params = {
        corpus: ELEMS.corpus.list.value,
        cqp: ELEMS.search.string.value,
        num: NUM_HITS,
    };
    call_api('query', params, show_search_results);
}


const STATE = {
    hits: null,
    start: null,
    end: null,
};


function navigate(nav) {
    if (!STATE.hits) return;
    let start = 0;
    if (nav === 'prev') {
        if (STATE.start >= NUM_HITS)
            start = STATE.start - NUM_HITS;
    }
    if (nav === 'next') {
        if (STATE.start < STATE.hits - NUM_HITS)
            start = STATE.start + NUM_HITS;
        else
            start = STATE.start;
    }
    if (nav === 'last') {
        if (STATE.hits > NUM_HITS)
            start = STATE.hits - NUM_HITS;
    }
    let params = {
        corpus: ELEMS.corpus.list.value,
        cqp: ELEMS.search.string.value,
        start: start,
        num: NUM_HITS,
    };
    call_api('query', params, show_search_results);
}


function show_search_results(response) {
    STATE.hits = response.hits; STATE.start = response.start; STATE.end = response.end;
    ELEMS.search.info.innerHTML = `
        Found ${response.hits.toLocaleString(LOCALE)} matches,
        showing n:o ${response.start.toLocaleString(LOCALE)}&ndash;${response.end.toLocaleString(LOCALE)}
        (completed in ${response.time.toLocaleString(LOCALE,{maximumFractionDigits:2})} s)
    `;
    ELEMS.search.results.innerHTML = '';
    let n = response.start;
    for (let line of response.kwic) {
        ELEMS.search.results.innerHTML +=
            '<tr><td class="prefix">' +
            line.tokens.map((token, i) =>
                (i===line.match.start ? '</td><td class="match">' : '') +
                show_token(token) +
                (i+1===line.match.end ? '</td><td class="suffix">' : '')
            ).join(' ') +
            '</td></tr>';
        n++;
    }
    let container = ELEMS.search.container;
    let match = document.querySelector('.match');
    scrollbar = match.offsetLeft + match.offsetWidth / 2 - container.offsetWidth / 2;
    ELEMS.search.container.scrollLeft = scrollbar;
}


function show_token(token) {
    let keys = Object.keys(token);
    let word = token[keys[0]];
    let title = keys.map((k) => `${k}: ${token[k]}`).join('&#13;');
    return `<span class="token" title="${title}">${word}</span>`;
}


function call_api(command, params, callback) {
    let queryparams = Object.keys(params).map((k) => `${k}=${params[k]}`).join('&');
    let url = `${API_DOMAIN}${command}?${queryparams}`;
    console.log(`API call: ${command} ? ${queryparams}`);
    let http = new XMLHttpRequest();
    if (!http) {
        error("Browser does not support HTTP Request", 500);
    }
    else {
        http.onreadystatechange = () => {
            if (http.readyState===4 || http.readyState==="complete") {
                let response = http.responseText;
                if (http.status >= 300) {
                    error(response, http.status);
                } else if (!response) {
                    error("Empty response form server (crash?)", 500);
                } else {
                    let obj;
                    try {
                        obj = JSON.parse(response);
                    } catch(e) {
                        error("JSON parsing problem", 400);
                    }
                    callback(obj, http.status);
                }
            }
        }
        http.open('GET', url, true);
        http.send();
    }
    return http;
}


function error(message, status) {
    if (status) {
        ELEMS.error.innerHTML = `<em>Error ${status}:</em> ${message}`
    } else {
        ELEMS.error.innerHTML = `<em>Error:</em> ${message}`
    }
}

