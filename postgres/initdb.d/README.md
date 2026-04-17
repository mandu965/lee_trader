This directory is mounted into `/docker-entrypoint-initdb.d` of the Postgres container.
Place SQL files here to create initial roles, schemas, or seed data.

Default container env:
- `POSTGRES_USER` / `POSTGRES_PASSWORD`
- `POSTGRES_DB`

Example (add as a new `.sql` file):
```sql
-- create read/write app role
CREATE ROLE app_rw LOGIN PASSWORD 'change-me';
GRANT CONNECT ON DATABASE lee_trader TO app_rw;
```
