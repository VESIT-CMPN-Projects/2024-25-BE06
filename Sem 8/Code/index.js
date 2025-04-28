const fs = require('fs');
const mysql = require('mysql2');
const csv = require('csv-parser');

const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: 'vedang123',
  database: 'hrdashboard',
});

function formatDate(originalDate) {
  if (!originalDate) return null; // handle empty dates
  const [month, day, year] = originalDate.split('/');
  if (!month || !day || !year) return null;
  return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
}

function cleanValue(value) {
  if (value === '' || value === '#N/A' || value === undefined) {
    return null;
  }
  return value;
}

const results = [];

fs.createReadStream('employee_data.csv')
  .pipe(csv())
  .on('data', (data) => {
    const formattedData = {
      employee_id: cleanValue(data['employee_id']),
      name: cleanValue(data['name']),
      position: cleanValue(data['position']),
      organizational_unit: cleanValue(data['organizational_unit']),
      ranks: cleanValue(data['ranks']),
      hire_date: formatDate(data['hire_date']),
      regularization_date: formatDate(data['regularization_date']),
      vacation_leave: cleanValue(data['vacation_leave']),
      sick_leave: cleanValue(data['sick_leave']),
      basic_pay_in_php: cleanValue(data['basic_pay_in_php']),
      employment_status: cleanValue(data['employment_status']),
      supervisor: cleanValue(data['supervisor']),
      password: cleanValue(data['password']),
      email: cleanValue(data['email'])
    };

    console.log(formattedData); // optional: check parsed data
    results.push(formattedData);
  })
  .on('end', () => {
    console.log('CSV file successfully processed');

    results.forEach((row) => {
      const sql = `
        INSERT INTO employee_data 
        (employee_id, name, position, organizational_unit, ranks, hire_date, regularization_date, vacation_leave, sick_leave, basic_pay_in_php, employment_status, supervisor, password, email)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `;

      const values = [
        row.employee_id,
        row.name,
        row.position,
        row.organizational_unit,
        row.ranks,
        row.hire_date,
        row.regularization_date,
        row.vacation_leave,
        row.sick_leave,
        row.basic_pay_in_php,
        row.employment_status,
        row.supervisor,
        row.password,
        row.email,
      ];

      connection.query(sql, values, (err) => {
        if (err) {
          console.error('Error inserting row:', err);
        } else {
          console.log('Row inserted successfully');
        }
      });
    });
  });
