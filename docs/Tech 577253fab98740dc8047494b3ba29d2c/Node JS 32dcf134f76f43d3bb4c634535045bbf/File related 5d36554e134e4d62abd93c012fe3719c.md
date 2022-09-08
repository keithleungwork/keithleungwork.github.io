# File related

---

## Browser Download

### Handle Blob/ file saving in browser

- Use the lib - [https://github.com/eligrey/FileSaver.js/](https://github.com/eligrey/FileSaver.js/)

```jsx
// Ref from https://medium.com/@fakiolinho/handle-blobs-requests-with-axios-the-right-way-bb905bdb1c04

// In fileSave.js
import FileSaver from 'file-saver';

export default (fileData, fileName) => FileSaver.saveAs(fileData, fileName);

// In <any API functions>.js
import fileSaver from './fileSaver';
import moment from "moment"

export async function downloadFile() {
	const blobData = await someApiRequest()
	const zip_file_name = moment().format("YYYYMMDD-hhmmss") + ".zip"
	await fileSaver(ran_res, zip_file_name);
}
```

---

## File Create/Read/Delete

### Write file

```jsx
const fs = require('fs');

const content = 'Some content!'

try {
  fs.writeFileSync('/Users/joe/test.txt', content)
  // file written successfully
} catch (err) {
  console.error(err)
}
```

### Read File

```jsx
const fs = require('fs')

css = fs.readFileSync(CSS_FILE_PATH, 'utf8')
```

### Create directory

```jsx
const fs = require('fs')

fs.mkdirSync("xxxx/xxxx")
```

### Remove directory

```jsx
const fs = require('fs')

fs.rmSync(dir_path, { recursive: true })
```

### Create symlink

```jsx
const fs = require('fs')

fs.symlinkSync(src_path, link_path, 'dir')
```

### Unlink symlink

```jsx
const fs = require('fs')

fs.unlinkSync(link_path)
```

---

## CSV file

Libs:

- Papaparse - [https://www.papaparse.com/docs](https://www.papaparse.com/docs)
- [https://github.com/adaltas/node-csv](https://github.com/adaltas/node-csv)

Example of reading a CSV ( with Papaparse )

```jsx
const Papa = require("papaparse")
const fs = require("fs")

// file path
let path = "/xxxx.csv"

// This is async way to read
function asyncWay() {
  let config = {
    complete: function(results, file) {
      console.log("Parsing complete:");
      console.log(results);
      console.log("-------------------")
      console.log(file);
    }
  }
  // Async here !!
  Papa.parse(
    fs.createReadStream(path),
    config)
}

// This is sync way
function syncWay(){
  const csv_str = fs.readFileSync(path, "utf8")
  const csvobj = Papa.parse(csv_str)  
  console.log(csvobj)
}

//asyncWay()
syncWay()

/* The result object looks like this
{
  data: [
    [ 'col_a', 'col_b', 'col_c' ],
    [
      '株式会社クリーク・アンド・リバー社',
      'CREEK & RIVER Co.,Ltd.',
      'カブシキガイシャクリークアンドリバーシャ'
    ],
    [ '' ]
  ],
  errors: [],
  meta: {
    delimiter: ',',
    linebreak: '\r\n',
    aborted: false,
    truncated: false,
    cursor: 84
  }
}
*/
```

---

## Path Manipulation

### Check path exist

```jsx
const fs = require('fs')

if (!fs.existsSync(entry_server_path)) {
	console.log("Not existed!")
}
```

### Path string concat

```jsx
const path = require("path")

path.join(__dirname, "../something.js")
```

### Path resolve

```jsx
const path = require("path")

path.resolve(path.join(dist_path, '/entry-server.css'))
```

### Get the most-right file name

```jsx
const path = require("path")

console.log(path.basename("/home/user/abc/def/tttt.txt"))
// tttt.txt
```

### Get the dir name

```jsx
const path = require("path")

console.log(path.dirname("/home/user/abc/def/tttt.txt"))
// /home/user/abc/def
console.log(path.dirname("../abc/def/tttt.txt"))
// ../abc/def
```

---

## Zip File

> Using lib: [https://www.archiverjs.com/docs/quickstart](https://www.archiverjs.com/docs/quickstart)
> 
- Zip some files: ***(In NodeJS)***
    
    > DO NOT rely on `archive.finalize()`, you should resolve in the “close/end” listener
    > 
    
    ```jsx
    // require modules
    const fs = require('fs');
    const archiver = require('archiver');
    
    // create a file to stream archive data to.
    const output = fs.createWriteStream(__dirname + '/example.zip');
    const archive = archiver('zip', {
      zlib: { level: 9 } // Sets the compression level.
    });
    
    // listen for all archive data to be written
    // 'close' event is fired only when a file descriptor is involved
    output.on('close', function() {
      console.log(archive.pointer() + ' total bytes');
      console.log('archiver has been finalized and the output file descriptor has closed.');
    });
    
    // This event is fired when the data source is drained no matter what was the data source.
    // It is not part of this library but rather from the NodeJS Stream API.
    // @see: https://nodejs.org/api/stream.html#stream_event_end
    output.on('end', function() {
      console.log('Data has been drained');
    });
    
    // good practice to catch warnings (ie stat failures and other non-blocking errors)
    archive.on('warning', function(err) {
      if (err.code === 'ENOENT') {
        // log warning
      } else {
        // throw error
        throw err;
      }
    });
    
    // good practice to catch this error explicitly
    archive.on('error', function(err) {
      throw err;
    });
    
    // pipe archive data to the file
    archive.pipe(output);
    
    // append a file from stream
    const file1 = __dirname + '/file1.txt';
    archive.append(fs.createReadStream(file1), { name: 'file1.txt' });
    
    // append a file from string
    archive.append('string cheese!', { name: 'file2.txt' });
    
    // append a file from buffer
    const buffer3 = Buffer.from('buff it!');
    archive.append(buffer3, { name: 'file3.txt' });
    
    // append a file
    archive.file('file1.txt', { name: 'file4.txt' });
    
    // append files from a sub-directory and naming it `new-subdir` within the archive
    archive.directory('subdir/', 'new-subdir');
    
    // append files from a sub-directory, putting its contents at the root of archive
    archive.directory('subdir/', false);
    
    // append files from a glob pattern
    archive.glob('file*.txt', {cwd:__dirname});
    
    // finalize the archive (ie we are done appending files but streams have to finish yet)
    // 'close', 'end' or 'finish' may be fired right after calling this method so register to them beforehand
    archive.finalize();
    ```