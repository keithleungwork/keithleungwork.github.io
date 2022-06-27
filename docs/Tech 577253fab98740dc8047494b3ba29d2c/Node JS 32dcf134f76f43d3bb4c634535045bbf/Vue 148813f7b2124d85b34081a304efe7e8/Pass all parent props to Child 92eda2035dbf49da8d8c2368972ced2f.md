# Pass all parent props to Child

It is common to make a wrapper component that just pass all parent props to child component.

You can use the var `$props` or `$attrs`

```jsx
<template>
  <!-- $attrs will include all props from parent EXCEPT those defined in props in this component -->
	<!-- i.e. contains all props, EXCEPT "excludeme" --> 
  <child-comp-1 v-bind="$attrs" />
	<!-- $props only contains the field: "excludeme" -->
  <child-comp-2 v-bind="$props" />
</template>

<script>
import ChildComp1 from "xxxxxxx"
import ChildComp2 from "xxxxxxx"

export default {
  name: "Wrapper",
  components: {
		ChildComp1,
		ChildComp2
  },
  props: {
    excludeme: {
      default: 1
    }
  },
}
</script>
```