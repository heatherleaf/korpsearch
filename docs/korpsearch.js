
const API_DOMAIN = "https://korpsearch.cse.chalmers.se:8000/";
const LOCALE = "en-US";

const ELEMS = {
    corpus: null,
    search: null,
    navigate: null,
    error: null,
};

const search_examples = {
    'English (BNC)': [
        '[pos="ART"] [lemma="big"] [pos="SUBST"]',
        '[lemma="be"] [pos="ART"] [pos="ADJ"] [pos="SUBST"]',
        '[pos="ADJ"] [lemma="cut" pos="VERB"]',
    ],
    'Swedish': [
        '[pos="DT"] [word="stort"] [pos="NN"]',
        '[word="är"] [pos="DT"] [pos="JJ"] [pos="NN"]',
        '[pos="NN"] [word="händer" pos!="NN"]',
    ],
};

const NUM_HITS = 20;       // N:o hits to show
const GROUP_BY = 'word';   // Feature to group by for statistics
const SAMPLING = 10_000;   // Max n:o results to sample from, for statistics

function showNum(num, digits = 0) {
    return num.toLocaleString(LOCALE, {minimumFractionDigits: digits, maximumFractionDigits: digits})
}


window.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    ELEMS.corpus = {
        list: document.getElementById('corpus-list'),
        info: document.getElementById('corpus-info'),
    };
    ELEMS.query = {
        string: document.getElementById('search-string'),
        examples: document.getElementById('search-examples'),
        search: document.getElementById('search-button'),
        count: document.getElementById('count-button'),
        info: document.getElementById('search-info'),
        results: document.getElementById('results-container'),
    },
    ELEMS.navigate = {
        container: document.getElementById('navigation-container'),
        buttons: {
            first: document.getElementById('navigate-first'),
            prev: document.getElementById('navigate-prev'),
            next: document.getElementById('navigate-next'),
            last: document.getElementById('navigate-last'),
        },
    }
    ELEMS.error = document.getElementById('error');

    call_api('info', {}, (response) => {
        ELEMS.navigate.container.classList.add("hidden");
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
    ELEMS.query.examples.add(opt);
    for (let corpus in search_examples) {
        let group = document.createElement('optgroup');
        group.label = corpus;
        for (let example of search_examples[corpus]){
            let opt = document.createElement('option');
            opt.text = opt.value = example;
            group.appendChild(opt);
        }
        ELEMS.query.examples.add(group);
    }

    ELEMS.corpus.list.addEventListener('change', select_corpus);
    ELEMS.query.examples.addEventListener('change', select_example);
    ELEMS.query.search.addEventListener('click', search_corpus);
    ELEMS.query.count.addEventListener('click', count_corpus);

    for (let nav in ELEMS.navigate.buttons) {
        ELEMS.navigate.buttons[nav].addEventListener('click', () => navigate(nav));
    }

    ELEMS.query.string.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault(); // Cancel the default action, if needed
            ELEMS.query.search.click();
        }
    });
}


function clear_output() {
    ELEMS.corpus.info.innerHTML = ELEMS.query.info.innerHTML = ELEMS.query.results.innerHTML = ELEMS.error.innerHTML = '';
}


function select_corpus() {
    clear_output();
    let corpus = ELEMS.corpus.list.value;
    call_api('corpus_info', {corpus: corpus}, (response) => {
        ELEMS.navigate.container.classList.add("hidden");
        let html = "";
        for (const corpus of Object.values(response.corpora)) {
            html += `<p>
                <strong>${corpus.info.Name}</strong>:
                ${showNum(corpus.info.Size)} tokens;
                ${showNum(corpus.info.Sentences)} sentences;
                ${showNum(corpus.info.Indexes.length)} compiled indexes;
                ${showNum(corpus.attrs.p.length)} features (<em>${corpus.attrs.p.join('</em>, <em>')}</em>).
            </p>`;
        }
        ELEMS.corpus.info.innerHTML = html;
    });
}


function select_example() {
    let examples = ELEMS.query.examples;
    if (examples.value) {
        ELEMS.query.string.value = examples.value;
        ELEMS.query.string.focus();
    }
    examples.selectedIndex = 0;
}


function search_corpus() {
    let params = {
        corpus: ELEMS.corpus.list.value,
        cqp: ELEMS.query.string.value,
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
        cqp: ELEMS.query.string.value,
        start: start,
        num: NUM_HITS,
    };
    console.log(STATE, params)
    call_api('query', params, show_search_results);
}


function show_search_results(response) {
    STATE.hits = response.hits; STATE.start = response.start; STATE.end = response.end;
    ELEMS.query.info.innerHTML = `
        Found ${showNum(response.hits)} matches,
        showing n:o ${showNum(response.start)}&ndash;${showNum(response.end)}
        (completed in ${showNum(response.time, 2)} s)
    `;
    ELEMS.navigate.container.classList.toggle("hidden", response.hits <= NUM_HITS);
    let html = '';
    let n = response.start;
    for (let line of response.kwic) {
        html +=
            '<tr><td class="prefix">' +
            line.tokens.map((token, i) =>
                (i===line.match.start ? '</td><td class="match">' : '') +
                show_token(token) +
                (i+1===line.match.end ? '</td><td class="suffix">' : '')
            ).join(' ') +
            '</td></tr>';
        n++;
    }
    let container = ELEMS.query.results;
    container.innerHTML = `<table id="query-results">${html}</table>`;
    let match = document.querySelector('.match');
    scrollbar = match.offsetLeft + match.offsetWidth / 2 - container.offsetWidth / 2;
    ELEMS.query.results.scrollLeft = scrollbar;
}


function show_token(token) {
    let keys = Object.keys(token);
    let word = token[keys[0]];
    let title = keys.map((k) => `${k}: ${token[k]}`).join('&#13;');
    return `<span class="token" title="${title}">${word}</span>`;
}


function count_corpus() {
    let params = {
        corpus: ELEMS.corpus.list.value,
        cqp: ELEMS.query.string.value,
        group_by: GROUP_BY,
        num: NUM_HITS,
        sampling: SAMPLING,
    };
    call_api('count', params, show_count_results);
}


function show_count_results(response){
    ELEMS.query.info.innerHTML = `
        Found ${showNum(response.count)} different groups,
        (completed in ${showNum(response.time, 2)} s)
    `;
    ELEMS.navigate.container.classList.add("hidden");
    const corpora = ["∑"];
    const stats = [response.combined];
    for (const crp in response.corpora) {
        corpora.push(crp);
        stats.push(response.corpora[crp]);
    }
    const table = {};
    for (let n = 0; n < stats.length; n++) {
        (table["∑"] ??= []).push(
            stats[n].sums.absolute,
            showNum(stats[n].sums.relative, 1),
            "",
        );
            for (const row of stats[n].rows) {
            const hdr = row.value[GROUP_BY].join(" ");
            (table[hdr] ??= []).push(
                row.absolute,
                showNum(row.relative, 1),
                showNum(100 * row.absolute / stats[n].sums.absolute, 1) + "%",
            );
        }
        for (const hdr in table) {
            while (table[hdr].length <= n) table[hdr].push("");
        }
    }
    let html = '';
    const row_hdrs = Object.keys(table);
    row_hdrs.sort((a,b) => table[b][0] - table[a][0]);
    row_hdrs.splice(NUM_HITS);
    let tblrow = '<th></th>';
    for (let n = 0; n < corpora.length; n++) {
        const sampling = 100 * stats[n].sums['sampled-hits'] / stats[n].sums['total-hits'];
        tblrow += `<th colspan="3" class="center">${corpora[n]} (${showNum(sampling, 1)}%)</th>`;
    }
    html += `<tr>${tblrow}</tr>`;
    tblrow = '<th></th>';
    for (let n = 0; n < corpora.length; n++) {
        tblrow += "<th>Hits</th><th>Hits/million</th><th>% of hits</th>";
    }
    html += `<tr>${tblrow}</tr>`;
    for (const crp of corpora) tblrow += `<th>${crp}</th>`;
    for (const hdr of row_hdrs) {
        tblrow = `<th>${hdr}</th>`;
        for (const n of table[hdr]) {
            tblrow += `<td>${typeof n === "number" ? showNum(n, 0) : n}</td>`
        };
        html += `<tr>${tblrow}</tr>`;
    }
    ELEMS.query.results.innerHTML = `<table id="count-results">${html}</table>`;
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

