# Knowledge Base Sources

Statements in each file were written from these references. Everything here is stable textbook, standards, or official-documentation material. Version-specific facts are dated in the text, e.g. "OWASP Top 10 2025".

## algorithms.txt
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms* (CLRS), 4th ed., MIT Press
- Sedgewick & Wayne, *Algorithms*, 4th ed. (algs4.cs.princeton.edu)
- MIT OpenCourseWare 6.006 Introduction to Algorithms
- Python docs: Sorting HOW TO (Timsort, stability)

## data_structures.txt
- CLRS (hash tables, trees, heaps, graphs)
- Sedgewick & Wayne, *Algorithms*
- Python docs: Built-in Types, `collections`, `heapq`; TimeComplexity wiki (wiki.python.org)
- Java SE API docs: `HashMap`, `ArrayList`, `LinkedList`

## databases.txt
- Silberschatz, Korth, Sudarshan, *Database System Concepts*, 7th ed.
- PostgreSQL documentation: Transaction Isolation, Indexes, MVCC
- MySQL documentation: InnoDB locking and isolation levels
- ISO/IEC 9075 (SQL standard)
- Codd (1970), "A Relational Model of Data for Large Shared Data Banks"
- Brewer's CAP theorem; Gilbert & Lynch (2002) formal proof

## networking.txt
- Kurose & Ross, *Computer Networking: A Top-Down Approach*
- IETF RFCs: 791 (IPv4), 8200 (IPv6), 9293 (TCP), 768 (UDP), 1034/1035 (DNS), 9110/9111/9112 (HTTP semantics, caching, HTTP/1.1), 9113 (HTTP/2), 9114 (HTTP/3), 9000 (QUIC), 8446 (TLS 1.3), 2131 (DHCP), 1918 (private addresses)
- ISO/IEC 7498-1 (OSI reference model)

## operating_systems.txt
- Silberschatz, Galvin, Gagne, *Operating System Concepts*, 10th ed.
- Arpaci-Dusseau, *Operating Systems: Three Easy Pieces* (ostep.org)
- Linux man-pages project (fork(2), exec(3), mmap(2), signal(7))
- POSIX.1 / IEEE Std 1003.1

## software_engineering.txt
- Gamma, Helm, Johnson, Vlissides, *Design Patterns* (1994)
- Martin, *Agile Software Development* (SOLID)
- Agile Manifesto (agilemanifesto.org, 2001)
- Scrum Guide (scrumguides.org, 2020)
- Semantic Versioning 2.0.0 (semver.org)
- Fielding (2000), dissertation defining REST

## testing.txt
- ISTQB Foundation Level Syllabus and Glossary
- IEEE/ISO/IEC 29119 (software testing)
- pytest documentation (fixtures, parametrize, markers)
- Fowler, "Mocks Aren't Stubs" and Test Pyramid articles (martinfowler.com)

## git_cicd.txt
- Git reference documentation (git-scm.com/docs)
- Chacon & Straub, *Pro Git*, 2nd ed. (git-scm.com/book)
- GitHub Actions documentation (docs.github.com/actions)
- Humble & Farley, *Continuous Delivery* (2010)

## web_development.txt
- MDN Web Docs (developer.mozilla.org): HTTP, CORS, cookies, DOM, event loop
- WHATWG HTML Living Standard; W3C CSS specifications
- ECMA-262 (ECMAScript)
- RFC 8259 (JSON), RFC 7519 (JWT), RFC 6749 (OAuth 2.0), RFC 6455 (WebSocket), RFC 6265 (cookies)
- OpenID Connect Core 1.0

## cloud_devops.txt
- NIST SP 800-145, *The NIST Definition of Cloud Computing*
- Kubernetes documentation (kubernetes.io/docs)
- Docker documentation (docs.docker.com)
- Open Container Initiative specifications
- AWS Shared Responsibility Model; AWS Well-Architected Framework
- Google SRE book, *Site Reliability Engineering* (sre.google)
- The Twelve-Factor App (12factor.net)

## cybersecurity.txt
- OWASP Top 10:2025 (owasp.org/Top10)
- OWASP Cheat Sheet Series
- NIST Cybersecurity Framework 2.0 (Feb 2024)
- NIST SP 800-207 (Zero Trust Architecture); SP 800-63B (Digital Identity / passwords)
- NIST FIPS 197 (AES), 180-4 (SHA), 203/204/205 (post-quantum, Aug 2024)
- MITRE ATT&CK and CVE/CWE programs; FIRST CVSS specification

## machine_learning.txt
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*
- Bishop, *Pattern Recognition and Machine Learning*
- scikit-learn User Guide (scikit-learn.org)
- Stanford CS229 lecture notes

## deep_learning.txt
- Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press (deeplearningbook.org)
- Stanford CS231n notes
- PyTorch documentation
- Original papers: Rumelhart et al. 1986 (backprop), Hochreiter & Schmidhuber 1997 (LSTM), He et al. 2015 (ResNet), Kingma & Ba 2014 (Adam), Srivastava et al. 2014 (dropout), Ioffe & Szegedy 2015 (batch norm)

## transformers.txt
- Vaswani et al. (2017), "Attention Is All You Need"
- Devlin et al. (2018), BERT; Radford et al. (2018/2019), GPT/GPT-2
- Hugging Face Transformers documentation
- Stanford CS224N lecture notes

## llms.txt
- Brown et al. (2020), "Language Models are Few-Shot Learners"
- Ouyang et al. (2022), InstructGPT
- Bai et al. (2022), Constitutional AI (Anthropic)
- Rafailov et al. (2023), DPO; Hu et al. (2021), LoRA; Dettmers et al. (2023), QLoRA
- Wei et al. (2022), chain-of-thought prompting
- Liu et al. (2023), "Lost in the Middle"
- Ji et al. (2023), "Survey of Hallucination in Natural Language Generation"
- Sennrich et al. (2016), BPE for NMT

## embeddings.txt
- Mikolov et al. (2013), Word2Vec
- Pennington, Socher, Manning (2014), GloVe
- Bojanowski et al. (2017), FastText
- Reimers & Gurevych (2019), Sentence-BERT
- sentence-transformers documentation (sbert.net), all-MiniLM-L6-v2 model card

## rag.txt
- Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"
- Thorne et al. (2018), FEVER dataset
- Gao et al. (2022), HyDE
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval* (retrieval metrics)
- RAGAS documentation

## vector_databases.txt
- FAISS documentation and wiki (github.com/facebookresearch/faiss)
- Johnson, Douze, Jégou (2017), "Billion-scale similarity search with GPUs"
- Malkov & Yashunin (2016/2018), HNSW
- Jégou, Douze, Schmid (2011), product quantization
- pgvector README (github.com/pgvector/pgvector)

## ai_safety.txt
- NIST AI RMF 1.0 (NIST AI 100-1, Jan 2023)
- NIST AI 600-1, Generative AI Profile (Jul 2024)
- EU AI Act, Regulation (EU) 2024/1689
- ISO/IEC 42001:2023
- OWASP Top 10 for LLM Applications 2025 (genai.owasp.org)
- MITRE ATLAS (atlas.mitre.org)
- Mitchell et al. (2019), Model Cards; Gebru et al. (2018), Datasheets for Datasets
- Lin et al. (2021), TruthfulQA
- Kleinberg et al. (2016); Chouldechova (2017), fairness impossibility results
- Carlini et al. (2021), training data extraction
