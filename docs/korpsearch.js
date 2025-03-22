
// const API_DOMAIN = "https://korpsearch.cse.chalmers.se:8000/";
const API_DOMAIN = "http://127.0.0.1:8000/";
const LOCALE = "en-US";

const ELEMS = {
    corpus: null,
    search: null,
    navigate: null,
    error: null,
};

const search_examples = [
    '[lemma="vara"] [pos="DT"] [pos="JJ"] [pos="NN"]',
    '[lemma="en"] [lemma="stor"] [pos="NN"]',
    '[pos="NN"] [word="händer" pos!="NN"]',
    '[word="hon|han"] [pos="VB"]',
    '[lemma="pojke|flicka"]',
];

const NUM_HITS = 20;       // N:o hits to show
const GROUP_BY = 'word';   // Feature to group by for statistics
const SAMPLING = 10_000;   // Max n:o results to sample from, for statistics

function showNum(num, digits = 0) {
    return num.toLocaleString(LOCALE, {minimumFractionDigits: digits, maximumFractionDigits: digits})
}


window.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    ELEMS.corpus = {
        table: document.getElementById('corpora-table'),
    };
    ELEMS.query = {
        string: document.getElementById('search-string'),
        examples: document.getElementById('search-examples'),
        search: document.getElementById('search-button'),
        type: document.getElementById('search-type'),
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
        call_api('corpus_info', {corpus: response.corpora.sort().join(",")}, (response) => {
            ELEMS.navigate.container.classList.add("hidden");
            let html = `
            <tr><th class="center">Corpus</th><th class="right">Tokens</th><th class="right">Sentences</th>
            <th class="center">Compiled indexes</th><th class="left">Features</th></tr>
            `;
            for (const corpus of Object.values(response.corpora).toSorted()) {
                html += `
                <tr><th class="left"><label><input type="checkbox" name="corpus" value="${corpus.info.Name}"/> ${corpus.info.Name}</label></th>
                <td class="right">${showNum(corpus.info.Size)}</td>
                <td class="right">${showNum(corpus.info.Sentences)}</td>
                <td class="center">${showNum(corpus.info.Indexes.length)}</td>
                <td class="left"><em>${corpus.attrs.p.join('</em>, <em>')}</em></td>
                </tr>`;
            }
            ELEMS.corpus.table.innerHTML = html;
            for (const checkbox of ELEMS.corpus.table.querySelectorAll('input')) {
                checkbox.addEventListener('click', search_corpus);
            }
            ELEMS.corpus.table.querySelector('input').checked = true;
        });
    });

    let opt = document.createElement('option');
    opt.text = '(try an example search)'
    ELEMS.query.examples.add(opt);
    for (const example of search_examples) {
        let opt = document.createElement('option');
        opt.text = opt.value = example;
        ELEMS.query.examples.appendChild(opt);
    }

    ELEMS.query.examples.addEventListener('change', select_example);
    ELEMS.query.search.addEventListener('click', search_corpus);
    for (const radio of ELEMS.query.type.querySelectorAll('input[type=radio]')) {
        radio.addEventListener('click', search_corpus);
    }

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


function get_selected_corpora() {
    return [...ELEMS.corpus.table.querySelectorAll('input[type=checkbox]:checked')].map((c) => c.value);
}


function get_query_type() {
    return ELEMS.query.type.querySelector('input[type=radio]:checked').value;
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
    const corpora = get_selected_corpora();
    const query = ELEMS.query.string.value.trim();
    console.log(query, corpora)
    if (corpora.length === 0 || !query) return;
    const params = {
        corpus: corpora.join(","),
        cqp: query,
        group_by: GROUP_BY,
        num: NUM_HITS,
        sampling: SAMPLING,
    };
    if (get_query_type() == "qwic") {
        call_api('query', params, show_search_results);
    } else if (get_query_type() == "statistics") {
        call_api('count', params, show_count_results);
    }
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
        corpus: get_selected_corpora().join(","),
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


function show_count_results(response){
    ELEMS.query.info.innerHTML = `
        Found ${showNum(response.count)} different groups
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
            while (table[hdr].length < 3*(n+1)) table[hdr].push("");
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

