variable "name_prefix"        { type = string }
variable "visibility_timeout" { type = number; default = 300 }
variable "max_receive_count"  { type = number; default = 5 }
variable "tags"               { type = map(string); default = {} }
