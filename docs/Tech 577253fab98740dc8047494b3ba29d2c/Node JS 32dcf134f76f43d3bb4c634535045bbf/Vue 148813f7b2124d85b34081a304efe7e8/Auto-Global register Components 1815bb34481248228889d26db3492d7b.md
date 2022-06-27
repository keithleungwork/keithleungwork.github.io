# Auto-Global register Components

For some cases, you want to globally / automatically import a bunch of components in a directory.

You can follow the [guide here](https://v2.vuejs.org/v2/guide/components-registration.html#Automatic-Global-Registration-of-Base-Components)

> Remember that **global registration must take place before the root Vue instance is created (with `new Vue`)**. **[Here’s an example](https://github.com/chrisvfritz/vue-enterprise-boilerplate/blob/master/src/components/_globals.js)** of this pattern in a real project context.
> 

```jsx
import Vue from 'vue'
import upperFirst from 'lodash/upperFirst'
import camelCase from 'lodash/camelCase'

const requireComponent = require.context(
  // The relative path of the components folder
  './components',
  // Whether or not to look in subfolders
  false,
  // The regular expression used to match base component filenames
  /Base[A-Z]\w+\.(vue|js)$/
)

requireComponent.keys().forEach(fileName => {
  // Get component config
  const componentConfig = requireComponent(fileName)

  // Get PascalCase name of component
  const componentName = upperFirst(
    camelCase(
      // Gets the file name regardless of folder depth
      fileName
        .split('/')
        .pop()
        .replace(/\.\w+$/, '')
    )
  )

  // Register component globally
  Vue.component(
    componentName,
    // Look for the component options on `.default`, which will
    // exist if the component was exported with `export default`,
    // otherwise fall back to module's root.
    componentConfig.default || componentConfig
  )
})
```

Also, can refer to the [real world example here](https://github.com/bencodezen/vue-enterprise-boilerplate/blob/main/src/components/_globals.js)