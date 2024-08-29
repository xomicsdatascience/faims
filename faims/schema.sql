PRAGMA foreign_keys = ON;

CREATE TABLE queue (
    id_peptides TEXT UNIQUE NOT NULL PRIMARY KEY,
    job_name VARCHAR(200),
    status TEXT,
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE results (
    id_result TEXT UNIQUE NOT NULL PRIMARY KEY,
    id_peptides TEXT NOT NULL,
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_peptides) REFERENCES queue(id_peptides)
);